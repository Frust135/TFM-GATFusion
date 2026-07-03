from pandas import read_csv


def load_expression(file_path, library_type="unstranded"):
    """Return a (gene_id, count) DataFrame for the requested library strandedness.

    Returns:
        DataFrame with columns ['gene_id', 'count'].
    """
    columns = ["gene_id", "unstranded", "stranded_forward", "stranded_reverse"]
    df = read_csv(file_path, sep="\t", names=columns)
    df = df[~df["gene_id"].str.startswith("N_")].copy()

    if library_type not in ("unstranded", "stranded_forward", "stranded_reverse"):
        raise ValueError(f"Invalid library_type '{library_type}'. " "Choose from: 'unstranded', 'stranded_forward', 'stranded_reverse'.")

    df_out = df[["gene_id", library_type]].rename(columns={library_type: "count"}).reset_index(drop=True)
    print(f"Loaded {len(df_out)} genes with expression data.")
    return df_out
