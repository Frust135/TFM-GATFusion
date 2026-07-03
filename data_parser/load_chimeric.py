import re

from pandas import read_csv

_CIGAR_M_RE = re.compile(r"(\d+)M")


def _cigar_match_length(cigar):
    """Return the total number of matched bases"""
    if not isinstance(cigar, str) or not cigar:
        return 0
    return sum(int(n) for n in _CIGAR_M_RE.findall(cigar))


def load_chimeric(filepath, min_split_reads=2):
    """Return a DataFrame of chimeric junctions with a minimum number of supporting reads.
    
    Columns returned:
        chr_donor, brkpt_donor, strand_donor,
        chr_acceptor, brkpt_acceptor, strand_acceptor,
        junction_type, num_split_reads,
        repeat_left_len, repeat_right_len,
        max_balanced_anchor
    """
    columns_star = [
        "chr_donor",
        "brkpt_donor",
        "strand_donor",
        "chr_acceptor",
        "brkpt_acceptor",
        "strand_acceptor",
        "junction_type",
        "repeat_left_len",
        "repeat_right_len",
        "read_name",
        "aln_1_start",
        "aln_1_cigar",
        "aln_2_start",
        "aln_2_cigar",
    ]
    df = read_csv(filepath, sep="\t", names=columns_star, comment="#")
    df["_left_anchor"] = df["aln_1_cigar"].apply(_cigar_match_length)
    df["_right_anchor"] = df["aln_2_cigar"].apply(_cigar_match_length)
    df["_balanced_anchor"] = df[["_left_anchor", "_right_anchor"]].min(axis=1)

    df_edges = (
        df.groupby(["chr_donor", "brkpt_donor", "strand_donor", "chr_acceptor", "brkpt_acceptor", "strand_acceptor"])
        .agg(
            num_split_reads=("read_name", "count"),
            junction_type=("junction_type", "first"),
            repeat_left_len=("repeat_left_len", "mean"),
            repeat_right_len=("repeat_right_len", "mean"),
            max_balanced_anchor=("_balanced_anchor", "max"),
        )
        .reset_index()
    )
    df_edges = df_edges[df_edges["num_split_reads"] >= min_split_reads].sort_values("num_split_reads", ascending=False).reset_index(drop=True)
    print(f"Loaded {len(df_edges)} chimeric junctions with >= {min_split_reads} split reads.")
    return df_edges
