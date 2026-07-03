from pandas import read_csv


def load_gene_annotations(gtf_filepath):
    """Return a DataFrame with one row per gene from a GTF file.

    Columns returned:
        chr, start, end, strand, gene_id, gene_name, gene_biotype, gene_length
    """
    columns = ["chr", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    df_gtf = read_csv(gtf_filepath, sep="\t", comment="#", names=columns)

    df_genes = df_gtf[df_gtf["feature"] == "gene"].copy()
    df_genes["gene_id"] = df_genes["attribute"].str.extract(r'gene_id "([^"]+)"')
    df_genes["gene_name"] = df_genes["attribute"].str.extract(r'gene_name "([^"]+)"')
    df_genes["gene_biotype"] = df_genes["attribute"].str.extract(r'gene_biotype "([^"]+)"')
    df_genes["gene_length"] = df_genes["end"] - df_genes["start"] + 1

    return df_genes[["chr", "start", "end", "strand", "gene_id", "gene_name", "gene_biotype", "gene_length"]].reset_index(drop=True)


def load_exon_annotations(gtf_filepath):
    """Return a DataFrame with one row per exon from a GTF file.

    Columns returned:
        chr, start, end, gene_id
    """
    columns = ["chr", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    df_gtf = read_csv(gtf_filepath, sep="\t", comment="#", names=columns)

    df_exons = df_gtf[df_gtf["feature"] == "exon"].copy()
    df_exons["gene_id"] = df_exons["attribute"].str.extract(r'gene_id "([^"]+)"')

    return df_exons[["chr", "start", "end", "gene_id"]].reset_index(drop=True)
