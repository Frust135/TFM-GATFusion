"""Download a reference genome assembly + annotation and build a STAR index.

Python port of download_references.sh from Arriba.

Usage:
    python scripts/download_references.py ASSEMBLY+ANNOTATION
    python scripts/download_references.py GRCh38+GENCODE38

Run with an unknown combination to print all valid ones. THREADS and
SJDBOVERHANG env vars (or --threads/--sjdb_overhang) control the STAR
genomeGenerate call.
"""

import argparse
import gzip
import io
import os
import re
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ASSEMBLIES = {
    "hs37d5": "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/phase2_reference_assembly_sequence/hs37d5.fa.gz",
    "hg19": "http://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/chromFa.tar.gz",
    "GRCh37": "http://ftp.ensembl.org/pub/grch37/release-87/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz",
    "hg38": "http://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.chromFa.tar.gz",
    "GRCh38": "http://ftp.ensembl.org/pub/release-93/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
    "mm10": "http://hgdownload.cse.ucsc.edu/goldenpath/mm10/bigZips/chromFa.tar.gz",
    "GRCm38": "http://ftp.ensembl.org/pub/release-99/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna.primary_assembly.fa.gz",
    "mm39": "http://hgdownload.cse.ucsc.edu/goldenpath/mm39/bigZips/mm39.chromFa.tar.gz",
    "GRCm39": "http://ftp.ensembl.org/pub/release-104/fasta/mus_musculus/dna/Mus_musculus.GRCm39.dna.primary_assembly.fa.gz",
}

ANNOTATIONS = {
    "GENCODE19": "http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz",
    "RefSeq_hg19": "http://hgdownload.cse.ucsc.edu/goldenpath/hg19/database/refGene.txt.gz",
    "ENSEMBL87": "http://ftp.ensembl.org/pub/grch37/release-87/gtf/homo_sapiens/Homo_sapiens.GRCh37.87.chr.gtf.gz",
    "GENCODE38": "http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.annotation.gtf.gz",
    "RefSeq_hg38": "http://hgdownload.cse.ucsc.edu/goldenpath/hg38/database/refGene.txt.gz",
    "ENSEMBL104": "http://ftp.ensembl.org/pub/release-104/gtf/homo_sapiens/Homo_sapiens.GRCh38.104.chr.gtf.gz",
    "GENCODEM25": "http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz",
    "RefSeq_mm10": "http://hgdownload.cse.ucsc.edu/goldenpath/mm10/database/refGene.txt.gz",
    "GENCODEM27": "http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M27/gencode.vM27.annotation.gtf.gz",
    "RefSeq_mm39": "http://hgdownload.cse.ucsc.edu/goldenpath/mm39/database/refGene.txt.gz",
}

COMBINATIONS = {
    "hs37d5+GENCODE19": "hs37d5+GENCODE19",
    "hs37d5+RefSeq": "hs37d5+RefSeq_hg19",
    "hs37d5+ENSEMBL87": "hs37d5+ENSEMBL87",
    "hg19+GENCODE19": "hg19+GENCODE19",
    "hg19+RefSeq": "hg19+RefSeq_hg19",
    "hg19+ENSEMBL87": "hg19+ENSEMBL87",
    "GRCh37+GENCODE19": "GRCh37+GENCODE19",
    "GRCh37+RefSeq": "GRCh37+RefSeq_hg19",
    "GRCh37+ENSEMBL87": "GRCh37+ENSEMBL87",
    "hg38+GENCODE38": "hg38+GENCODE38",
    "hg38+RefSeq": "hg38+RefSeq_hg38",
    "hg38+ENSEMBL104": "hg38+ENSEMBL104",
    "GRCh38+GENCODE38": "GRCh38+GENCODE38",
    "GRCh38+RefSeq": "GRCh38+RefSeq_hg38",
    "GRCh38+ENSEMBL104": "GRCh38+ENSEMBL104",
    "GRCm38+GENCODEM25": "GRCm38+GENCODEM25",
    "GRCm38+RefSeq": "GRCm38+RefSeq_mm10",
    "mm10+GENCODEM25": "mm10+GENCODEM25",
    "mm10+RefSeq": "mm10+RefSeq_mm10",
    "GRCm39+GENCODEM27": "GRCm39+GENCODEM27",
    "GRCm39+RefSeq": "GRCm39+RefSeq_mm39",
    "mm39+GENCODEM27": "mm39+GENCODEM27",
    "mm39+RefSeq": "mm39+RefSeq_mm39",
}
for _combo, _target in list(COMBINATIONS.items()):
    _assembly, _annotation = _combo.split("+", 1)
    _t_assembly, _t_annotation = _target.split("+", 1)
    COMBINATIONS[f"{_assembly}viral+{_annotation}"] = f"{_t_assembly}viral+{_t_annotation}"

VIRAL_GENOMES_FILE = "RefSeq_viral_genomes_v2.5.1.fa.gz"


# --- Streaming download + decompression ---


def _open_url(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req)


def _decompressed_lines(resp, url):
    """Yield decoded text lines from a URL response, transparently handling .tar.gz / .gz / plain."""
    if url.endswith(".tar.gz"):
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                with tar.extractfile(member) as fh:
                    yield from io.TextIOWrapper(fh, encoding="utf-8")
    elif url.endswith(".gz"):
        with gzip.GzipFile(fileobj=resp) as gz:
            yield from io.TextIOWrapper(gz, encoding="utf-8")
    else:
        yield from io.TextIOWrapper(resp, encoding="utf-8")


def _filter_viral(lines):
    """Drop RefSeq/NCBI accession contigs (headers matching '>NC_' or '>AC_') and their sequence."""
    keep = True
    for line in lines:
        if line.startswith(">"):
            contig = line.split()[0]
            keep = not (contig.startswith(">NC_") or contig.startswith(">AC_"))
        if keep:
            yield line


# --- Assembly ---


def _download_assembly(url, dest_path, viral):
    print(f"Downloading assembly: {url}")
    with _open_url(url) as resp:
        lines = _decompressed_lines(resp, url)
        if viral:
            lines = _filter_viral(lines)
        with open(dest_path, "w", encoding="utf-8") as out:
            out.writelines(lines)


def _append_viral_genomes(dest_path, script_dir):
    candidates = [script_dir / VIRAL_GENOMES_FILE, script_dir / "database" / VIRAL_GENOMES_FILE]
    viral_fa = next((p for p in candidates if p.exists()), None)
    if viral_fa is None:
        raise FileNotFoundError(f"{VIRAL_GENOMES_FILE} not found next to the script or in its database/ subfolder — required for *viral combinations.")
    print("Appending RefSeq viral genomes")
    with gzip.open(viral_fa, "rt", encoding="utf-8") as src, open(dest_path, "a", encoding="utf-8") as out:
        for line in src:
            out.write(line)


def _fasta_uses_chr_prefix(fasta_path):
    with open(fasta_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">chr"):
                return True
    return False


def _maybe_index_fasta(fasta_path):
    try:
        result = subprocess.run(["samtools", "--version-only"], capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return
    if result.stdout.strip().startswith("1."):
        print("Indexing assembly")
        subprocess.run(["samtools", "faidx", str(fasta_path)], check=True)


# --- Annotation ---


def _min(a, b):
    return b if a > b else a


def _max(a, b):
    return b if a < b else a


def _genepred_to_gtf(lines):
    """Convert UCSC genePred rows (refGene.txt) to GTF exon/CDS records.

    Port of Arriba's awk script: trims the stop codon off "complete" CDS
    ends, disambiguates repeated transcript IDs, and emits one exon (+ one
    CDS, where coding) line per exon.
    """
    transcripts = {}
    records = []
    for line in lines:
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        name, chrom, strand = f[1], f[2], f[3]
        cds_start, cds_end = int(f[6]), int(f[7])
        exon_count = int(f[8])
        starts = [int(x) for x in f[9].rstrip(",").split(",")]
        ends = [int(x) for x in f[10].rstrip(",").split(",")]
        frames = [int(x) for x in f[15].rstrip(",").split(",")]
        gene_symbol = f[12]
        cds_start_stat, cds_end_stat = f[13], f[14]

        # Remove the stop codon from the CDS end annotated "complete" (cmpl).
        if strand == "-" and cds_start_stat == "cmpl" and (starts[0] != cds_start or (_min(ends[0], cds_end) - starts[0] + frames[0]) % 3 == 0):
            cds_start += 3
            for i in range(exon_count - 1):
                if ends[i] <= cds_start <= ends[i] + 2:
                    cds_start += starts[i + 1] - ends[i]
        if strand == "+" and cds_end_stat == "cmpl" and (ends[-1] != cds_end or (ends[-1] - _max(starts[-1], cds_start) + frames[-1]) % 3 == 0):
            cds_end -= 3
            for i in range(1, exon_count):
                if starts[i] - 2 <= cds_end <= starts[i]:
                    cds_end -= starts[i] - ends[i - 1]

        # Disambiguate repeated transcript IDs with a running suffix.
        transcripts[name] = transcripts.get(name, 0) + 1
        count = transcripts[name]
        gene_id = gene_symbol if count == 1 else f"{gene_symbol}_{count}"
        if count > 1:
            name = f"{name}_{count}"

        is_coding = "cmpl" in cds_start_stat  # matches awk's $14~/cmpl/ (also true for "incmpl")
        for i in range(exon_count):
            exon_number = i + 1 if strand == "+" else exon_count - i
            attributes = f'gene_id "{gene_id}"; transcript_id "{name}"; exon_number "{exon_number}"; exon_id "{name}.{exon_number}"; gene_name "{gene_symbol}";'
            records.append((chrom, "exon", starts[i] + 1, ends[i], strand, ".", attributes))
            if is_coding and cds_start <= ends[i] and cds_end >= starts[i]:
                records.append((chrom, "CDS", _max(cds_start, starts[i]) + 1, _min(cds_end, ends[i]), strand, frames[i], attributes))
    return records


_NUM_RE = re.compile(r"(\d+)")


def _natural_key(chrom):
    """Version-sort-ish key: numeric chunks compare as numbers, so chr2 < chr10."""
    return [(0, int(tok)) if tok.isdigit() else (1, tok) for tok in _NUM_RE.split(chrom) if tok]


def _gtf_lines_from_records(records):
    records = sorted(records, key=lambda r: (_natural_key(r[0]), r[2], r[3], r[1]))
    for chrom, feature, start, end, strand, frame, attributes in records:
        yield f"{chrom}\tRefSeq\t{feature}\t{start}\t{end}\t.\t{strand}\t{frame}\t{attributes}\n"


def _apply_chr_convention(lines, chr_prefixed):
    """Match the assembly FASTA's chromosome naming (UCSC 'chr1' vs Ensembl '1')."""
    numeric_chrom = re.compile(r"^([12][0-9]|[1-9XY])")
    for line in lines:
        if chr_prefixed:
            if line.startswith("MT"):
                line = "chrM" + line[2:]
            else:
                line = numeric_chrom.sub(lambda m: "chr" + m.group(1), line, count=1)
        else:
            if line.startswith("chrM"):
                line = "MT" + line[4:]
            elif line.startswith("chr"):
                line = line[3:]
        yield line


def _download_annotation(url, annotation_name, chr_prefixed, dest_path):
    print(f"Downloading annotation: {url}")
    with _open_url(url) as resp:
        lines = _decompressed_lines(resp, url)
        if "RefSeq" in annotation_name:
            lines = _gtf_lines_from_records(_genepred_to_gtf(lines))
        lines = _apply_chr_convention(lines, chr_prefixed)
        with open(dest_path, "w", encoding="utf-8") as out:
            out.writelines(lines)


# --- STAR index ---


def _build_star_index(assembly_label, annotation, fasta_path, gtf_path, threads, sjdb_overhang):
    index_dir = Path(f"STAR_index_{assembly_label}_{annotation}")
    index_dir.mkdir()
    subprocess.run(
        [
            "STAR",
            "--runMode",
            "genomeGenerate",
            "--genomeDir",
            str(index_dir),
            "--genomeFastaFiles",
            str(fasta_path),
            "--sjdbGTFfile",
            str(gtf_path),
            "--runThreadN",
            str(threads),
            "--sjdbOverhang",
            str(sjdb_overhang),
        ],
        check=True,
    )


# --- Entry point ---


def _parse_args():
    parser = argparse.ArgumentParser(description="Download a reference genome assembly + annotation and build a STAR index.")
    parser.add_argument("combination", help="ASSEMBLY+ANNOTATION, e.g. GRCh38+GENCODE38. Run with an unknown value to list all valid combinations.")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", 8)), help="STAR --runThreadN.")
    parser.add_argument("--sjdb_overhang", type=int, default=int(os.environ.get("SJDBOVERHANG", 250)), help="STAR --sjdbOverhang.")
    args = parser.parse_args()
    if args.combination not in COMBINATIONS:
        print(f"Unknown combination: {args.combination}", file=sys.stderr)
        print("Available assemblies and annotations:", file=sys.stderr)
        for combo in sorted(COMBINATIONS):
            print(combo, file=sys.stderr)
        sys.exit(1)
    return args


def main():
    args = _parse_args()

    target = COMBINATIONS[args.combination]
    assembly, annotation = target.split("+", 1)
    viral = assembly.endswith("viral")
    if viral:
        assembly = assembly[: -len("viral")]
    assembly_label = f"{assembly}viral" if viral else assembly

    fasta_path = Path(f"{assembly_label}.fa")
    _download_assembly(ASSEMBLIES[assembly], fasta_path, viral)
    if viral:
        _append_viral_genomes(fasta_path, Path(__file__).resolve().parent)
    _maybe_index_fasta(fasta_path)

    chr_prefixed = _fasta_uses_chr_prefix(fasta_path)
    gtf_path = Path(f"{annotation}.gtf")
    _download_annotation(ANNOTATIONS[annotation], annotation, chr_prefixed, gtf_path)

    print(f"\n[*] Building STAR index for {assembly_label} + {annotation}...")
    _build_star_index(assembly_label, annotation, fasta_path, gtf_path, args.threads, args.sjdb_overhang)
    print("Done.")


if __name__ == "__main__":
    main()
