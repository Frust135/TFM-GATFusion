from pandas import DataFrame
from pysam import AlignmentFile


def load_discordant_pairs(bam_filepath, threads=1):
    """Return a DataFrame of individual inter-chromosomal discordant read pairs
    
    Columns returned:
        chr_A, pos_A, chr_B, pos_B
    """
    pairs = []

    with AlignmentFile(bam_filepath, "rb", threads=max(1, int(threads))) as bam:
        for read in bam:
            if read.is_unmapped or not read.is_paired or read.mate_is_unmapped:
                continue
            if not read.is_read1:
                continue
            chr_self = read.reference_name
            chr_mate = read.next_reference_name
            if chr_self and chr_mate and chr_self != chr_mate:
                pairs.append((chr_self, read.reference_start, chr_mate, read.next_reference_start))

    df = DataFrame(pairs, columns=["chr_A", "pos_A", "chr_B", "pos_B"])
    print(f"Discordant pairs loaded: {len(df)} inter-chromosomal read pairs.")
    return df
