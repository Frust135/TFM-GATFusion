import re

# Highly-expressed / repetitive gene families that generate large numbers of chimeric reads in most RNA-seq libraries.
_BLACKLIST_REGEXES = [
    re.compile(r"^RPL\d"),  # cytoplasmic ribosomal large subunit
    re.compile(r"^RPS\d"),  # cytoplasmic ribosomal small subunit
    re.compile(r"^MRPL\d"),  # mitochondrial ribosomal large subunit
    re.compile(r"^MRPS\d"),  # mitochondrial ribosomal small subunit
    re.compile(r"^MT-"),  # mitochondrially encoded
    re.compile(r"^HIST\d"),  # histones
    re.compile(r"^RNU\d"),  # small nuclear RNAs
    re.compile(r"^SNOR[AD]\d"),  # snoRNAs
    re.compile(r"^RN7S"),  # 7SK / 7SL
    re.compile(r"^RNA5"),  # 5S/5.8S rRNAs
    re.compile(r"^TRNA"),  # tRNAs
]


def is_blacklisted(gene_name):
    """Return True if the gene matches a known artefact family"""
    if not isinstance(gene_name, str) or not gene_name:
        return False
    return any(rx.match(gene_name) for rx in _BLACKLIST_REGEXES)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def _is_noncanonical_artifact(row, min_pct_canonical=0.1):
    """Detect inter-chromosomal exon-exon junctions with no canonical splice-site signal"""
    pct = row.get("pct_canonical", 1.0)
    if pct >= min_pct_canonical:
        return False
    if row.get("fusion_type") != "inter-chromosomal":
        return False
    return row.get("donor_region") == "exon" and row.get("acceptor_region") == "exon"


def _is_readthrough(row, max_dist):
    """Detect adjacent same-chromosome same-direction read-through events"""
    if row["chr_donor"] != row["chr_acceptor"]:
        return False
    if row["donor_gene"] == row["acceptor_gene"]:
        return False  # handled by intragenic filter
    if abs(int(row["brkpt_donor"]) - int(row["brkpt_acceptor"])) > max_dist:
        return False
    same_strand = row.get("strand_donor") == row.get("strand_acceptor")
    return bool(same_strand)


def apply_postprocess(
    df,
    *,
    drop_intragenic=True,
    drop_readthrough=True,
    drop_blacklisted=True,
    drop_noncanonical_artifacts=True,
    noncanonical_min_pct=0.1,
    readthrough_max_dist=100_000,
    annotate_only=False,
):
    """Annotate and filter the predicted fusions"""
    if df is None or df.empty:
        return df, df.iloc[0:0].copy() if df is not None else df

    out = df.copy()

    def _reason(row):
        reasons = []
        if drop_intragenic and row["donor_gene"] == row["acceptor_gene"]:
            reasons.append("intragenic")
        if drop_blacklisted and (is_blacklisted(row["donor_gene"]) or is_blacklisted(row["acceptor_gene"])):
            reasons.append("blacklist")
        if drop_readthrough and _is_readthrough(row, readthrough_max_dist):
            reasons.append("readthrough")
        if drop_noncanonical_artifacts and _is_noncanonical_artifact(row, noncanonical_min_pct):
            reasons.append("noncanonical_artifact")
        return ";".join(reasons)

    out["filter_reason"] = out.apply(_reason, axis=1)

    if annotate_only:
        return out, out.iloc[0:0].copy()

    keep_mask = out["filter_reason"] == ""
    df_kept = out[keep_mask].drop(columns=["filter_reason"]).reset_index(drop=True)
    df_dropped = out[~keep_mask].reset_index(drop=True)

    n_dropped = len(df_dropped)
    if n_dropped:
        by_reason = df_dropped["filter_reason"].value_counts().to_dict()
        breakdown = ", ".join(f"{k}={v}" for k, v in by_reason.items())
        print(f"[*] Post-processing: dropped {n_dropped} edge(s) ({breakdown}).")
    else:
        print("[*] Post-processing: no edges removed.")
    return df_kept, df_dropped
