import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def _build_gene_position_index(gtf_df):
    """Build a per-chromosome sorted index"""
    has_biotype = "gene_biotype" in gtf_df.columns
    index = {}
    for chrom, group in gtf_df.groupby("chr", sort=False):
        order = group["start"].values.argsort()
        starts = group["start"].values[order].astype(np.int64)
        ends = group["end"].values[order].astype(np.int64)
        gids = group["gene_id"].values[order]
        if has_biotype:
            coding = group["gene_biotype"].values[order] == "protein_coding"
        else:
            coding = np.ones(len(starts), dtype=bool)
        index[chrom] = (starts, ends, gids, coding)
    return index


def _lookup_genes_indexed(index, chrom, pos):
    """Return gene_ids overlapping (chrom, pos)"""
    entry = index.get(chrom)
    if entry is None:
        return []
    starts, ends, gids, coding = entry
    hi = np.searchsorted(starts, pos, side="right")
    if hi == 0:
        return []
    mask = ends[:hi] >= pos
    if not mask.any():
        return []
    if coding[:hi][mask].any():
        sel = mask & coding[:hi]
        return list(dict.fromkeys(gids[:hi][sel].tolist()))
    return list(dict.fromkeys(gids[:hi][mask].tolist()))


def _compute_tpm(reads_per_gene_df, gtf_df):
    """Calculate TPM from read counts and gene lengths from the GTF"""
    gene_length_map = gtf_df[["gene_id", "gene_length"]].drop_duplicates("gene_id").set_index("gene_id")["gene_length"]
    df = reads_per_gene_df.copy()
    df["gene_length"] = df["gene_id"].map(gene_length_map).fillna(1000)
    df["rpk"] = df["count"] / (df["gene_length"] / 1000.0)
    scale_factor = df["rpk"].sum() / 1e6
    df["tpm"] = df["rpk"] / scale_factor if scale_factor > 0 else 0.0
    return df


def _one_hot_junction(junction_type):
    """One-hot encode STAR junction type (0-3)"""
    vec = np.zeros(4)
    if 0 <= int(junction_type) < 4:
        vec[int(junction_type)] = 1.0
    return vec


def _encode_strand(donor, acceptor):
    """Three-class encoding of the donor/acceptor strand orientation"""
    if donor == "+" and acceptor == "-":
        return [1, 0, 0]  # canonical read-through
    if donor == "+" and acceptor == "+":
        return [0, 1, 0]  # tandem duplication
    if donor == "-" and acceptor == "-":
        return [0, 0, 1]  # inversion
    return [0, 0, 0]  # other / unknown


def _build_gene_exon_map(exon_df):
    """Build a mapping dict for genes based on their exon intervals"""
    gene_exons = {}
    if exon_df is None:
        return gene_exons
    for _, row in exon_df.iterrows():
        gene_exons.setdefault(row["gene_id"], []).append((row["start"], row["end"]))
    for gid in gene_exons:
        gene_exons[gid].sort()
    return gene_exons


def _breakpoint_in_exon(gene_exons, gene_id, pos):
    """Check if position falls within any exon of gene_id"""
    exons = gene_exons.get(gene_id, [])
    for start, end in exons:
        if start <= pos <= end:
            return 1.0
        if start > pos:
            break
    return 0.0


def _relative_position(gene_coords, gene_id, pos):
    """Return the relative position of position within gene_id scaled to [0, 1]"""
    coords = gene_coords.get(gene_id)
    if coords is None:
        return 0.5
    gene_start, gene_end = coords
    length = gene_end - gene_start + 1
    if length <= 0:
        return 0.5
    return max(0.0, min(1.0, (pos - gene_start) / length))


def create_graph(reads_per_gene_df, chimeric_df, discordant_df, gtf_df, mitelman_db=None, exon_df=None):
    """
    Create a graph from expression, chimeric, discordant, and annotation data.

    Node features (3 without Mitelman, 5 with):
        - log1p(TPM)                     z-score normalised
        - log1p(gene_length)             z-score normalised
        - is_protein_coding              1 if gene_biotype == protein_coding
        - [is_fusion_gene]               1 if gene appears in Mitelman  (optional)
        - [log1p(partner_count)]         unique partners in Mitelman    (optional)

    Edge features (29 without Mitelman, 31 with):
        - log1p(split_reads)                                     idx 0
        - log1p(discordant_pairs)       per-junction ±10kb    idx 1
        - split / (discordant + 1)       evidence ratio          idx 2
        - log1p(breakpoint distance)                             idx 3
        - log1p(TPM_donor / TPM_acceptor)                        idx 4
        - is_interchromosomal                                    idx 5
        - junction_type one-hot          4 dims                  idx 6-9
        - strand orientation             3 dims                  idx 10-12
        - log1p(repeat_left_len)                                 idx 13
        - log1p(repeat_right_len)                                idx 14

        - donor_in_exon                  1 if exonic breakpoint  idx 15
        - acceptor_in_exon               1 if exonic breakpoint  idx 16
        - donor_relative_pos             0-1 within gene body    idx 17
        - acceptor_relative_pos          0-1 within gene body    idx 18

        - log1p(FFPM*1000)               fusion fragments/M reads idx 19
        - log1p(max_balanced_anchor)     LDAS proxy              idx 20
        - log1p(promiscuity_donor)       # distinct partners     idx 21
        - log1p(promiscuity_acceptor)    # distinct partners     idx 22
        - is_read_through                same-chr same-strand adj idx 23

        - [is_known_pair]                                        idx 24
        - [log1p(recurrence)]                                    idx 25

        - log1p(num_junctions)           for this gene pair      idx 24/26
        - log1p(max_split_reads)         strongest junction      idx 25/27
        - frac_canonical_junctions       annotated splice sites  idx 26/28
        - log1p(total_chimeric_donor)    background noise        idx 27/29
        - log1p(total_chimeric_acceptor) background noise        idx 28/30
        - pct_canonical * both_exonic    interaction term        idx 29/31

    Args:
        reads_per_gene_df: DataFrame from load_expression.
        chimeric_df:       DataFrame from load_chimeric.
        discordant_df:     DataFrame from load_discordant_pairs.
        gtf_df:            DataFrame from load_gene_annotations.
        mitelman_db:       Optional MitelmanDB from load_mitelman.
        exon_df:           Optional DataFrame from load_exon_annotations.
    """

    # -------------------------------------------------------------------------
    # Nodes
    # -------------------------------------------------------------------------
    df_expr = _compute_tpm(reads_per_gene_df, gtf_df)
    df_expr = df_expr[df_expr["tpm"] > 0].reset_index(drop=True)

    unique_genes = df_expr["gene_id"].unique()
    gene_to_node_id = {g: i for i, g in enumerate(unique_genes)}

    tpm_vals = np.log1p(df_expr["tpm"].values)
    length_vals = np.log1p(df_expr["gene_length"].values)
    tpm_vals = (tpm_vals - tpm_vals.mean()) / (tpm_vals.std() + 1e-6)
    length_vals = (length_vals - length_vals.mean()) / (length_vals.std() + 1e-6)

    biotype_map = gtf_df[["gene_id", "gene_biotype"]].drop_duplicates("gene_id").set_index("gene_id")["gene_biotype"]
    is_protein_coding = np.array(
        [1.0 if biotype_map.get(g, "") == "protein_coding" else 0.0 for g in unique_genes],
        dtype=np.float32,
    )

    node_feature_cols = [tpm_vals, length_vals, is_protein_coding]

    if mitelman_db is not None:
        gene_id_to_name_tmp = gtf_df[["gene_id", "gene_name"]].drop_duplicates("gene_id").set_index("gene_id")["gene_name"]
        is_fusion_gene = np.array([1.0 if mitelman_db.is_fusion_gene(gene_id_to_name_tmp.get(g, "")) else 0.0 for g in unique_genes], dtype=np.float32)
        raw_partner_count = np.array([float(mitelman_db.partner_count(gene_id_to_name_tmp.get(g, ""))) for g in unique_genes], dtype=np.float32)
        node_feature_cols += [is_fusion_gene, np.log1p(raw_partner_count)]

    x = torch.tensor(np.stack(node_feature_cols, axis=1), dtype=torch.float)
    print(f"[*] Nodes: {len(unique_genes)} | Node features: {x.shape[1]}")

    # -------------------------------------------------------------------------
    # Breakpoint annotation lookups
    # -------------------------------------------------------------------------
    gene_exon_map = _build_gene_exon_map(exon_df)
    gene_coords = gtf_df[["gene_id", "start", "end"]].drop_duplicates("gene_id").set_index("gene_id").apply(lambda r: (r["start"], r["end"]), axis=1).to_dict()

    tpm_lookup = df_expr.set_index("gene_id")["tpm"].to_dict()

    # -------------------------------------------------------------------------
    # FFPM normalization factor
    # -------------------------------------------------------------------------
    if "count" in reads_per_gene_df.columns:
        total_mapped_reads = float(reads_per_gene_df["count"].sum())
    else:
        total_mapped_reads = 0.0
    ffpm_denom = max(total_mapped_reads / 1e6, 1e-6)

    # -------------------------------------------------------------------------
    # Gene strand lookup (for read-through detection)
    # -------------------------------------------------------------------------
    if "strand" in gtf_df.columns:
        gene_strand = gtf_df[["gene_id", "strand"]].drop_duplicates("gene_id").set_index("gene_id")["strand"].to_dict()
    else:
        gene_strand = {}
    gene_chrom = gtf_df[["gene_id", "chr"]].drop_duplicates("gene_id").set_index("gene_id")["chr"].to_dict()

    _READTHROUGH_MAX_DIST = 100_000

    def _is_read_through(g1, g2):
        """Return 1.0 if g1 and g2 are on the same chr+strand and within 100 kb"""
        if g1 == g2:
            return 0.0
        c1, c2 = gene_chrom.get(g1), gene_chrom.get(g2)
        if c1 is None or c2 is None or c1 != c2:
            return 0.0
        s1, s2 = gene_strand.get(g1), gene_strand.get(g2)
        if not s1 or not s2 or s1 != s2:
            return 0.0
        b1, b2 = gene_coords.get(g1), gene_coords.get(g2)
        if b1 is None or b2 is None:
            return 0.0
        dist = max(b1[0], b2[0]) - min(b1[1], b2[1])
        return 1.0 if dist <= _READTHROUGH_MAX_DIST else 0.0

    promiscuity = {}
    promiscuity_count = {}

    # -------------------------------------------------------------------------
    # Discordant-pair spatial index
    # -------------------------------------------------------------------------
    _DISC_WINDOW = 10_000

    _disc_index = {}  # (chr_d, chr_a) -> np array of (pos_d, pos_a)
    if not discordant_df.empty and "pos_A" in discordant_df.columns:
        chr_a_arr = discordant_df["chr_A"].values
        chr_b_arr = discordant_df["chr_B"].values
        pos_a_arr = discordant_df["pos_A"].values.astype(np.int64)
        pos_b_arr = discordant_df["pos_B"].values.astype(np.int64)
        pair_keys = pd.Series(list(zip(chr_a_arr, chr_b_arr))) if False else None
        tmp = pd.DataFrame(
            {
                "k1": list(zip(chr_a_arr, chr_b_arr)),
                "pa": pos_a_arr,
                "pb": pos_b_arr,
            }
        )
        for k, grp in tmp.groupby("k1", sort=False):
            arr = np.stack([grp["pa"].values, grp["pb"].values], axis=1)
            _disc_index.setdefault(k, []).append(arr)
            arr_rev = np.stack([grp["pb"].values, grp["pa"].values], axis=1)
            _disc_index.setdefault((k[1], k[0]), []).append(arr_rev)
        _disc_index = {k: np.vstack(v) for k, v in _disc_index.items()}

    def _count_discordant_near(chr_d, pos_d, chr_a, pos_a):
        """Count discordant pairs of a breakpoint"""
        arr = _disc_index.get((chr_d, chr_a))
        if arr is None or len(arr) == 0:
            return 0.0
        near = (np.abs(arr[:, 0] - pos_d) <= _DISC_WINDOW) & (np.abs(arr[:, 1] - pos_a) <= _DISC_WINDOW)
        return float(near.sum())

    # -------------------------------------------------------------------------
    # Features
    # -------------------------------------------------------------------------
    gene_pos_index = _build_gene_position_index(gtf_df)
    _bp_gene_cache = {}

    def _genes_at(chrom, pos):
        key = (chrom, int(pos))
        cached = _bp_gene_cache.get(key)
        if cached is None:
            cached = _lookup_genes_indexed(gene_pos_index, chrom, pos)
            _bp_gene_cache[key] = cached
        return cached

    gene_pair_stats = {}
    gene_total_split_reads = {}

    _chrom_d_col = chimeric_df["chr_donor"].values
    _brkpt_d_col = chimeric_df["brkpt_donor"].values
    _chrom_a_col = chimeric_df["chr_acceptor"].values
    _brkpt_a_col = chimeric_df["brkpt_acceptor"].values
    _split_col = chimeric_df["num_split_reads"].values.astype(np.float64)
    _jtype_col = chimeric_df["junction_type"].values

    for _i in range(len(chimeric_df)):
        donor_genes = _genes_at(_chrom_d_col[_i], _brkpt_d_col[_i])
        acceptor_genes = _genes_at(_chrom_a_col[_i], _brkpt_a_col[_i])

        split_reads_row = float(_split_col[_i])
        for _g in donor_genes:
            if _g in gene_to_node_id:
                _n = gene_to_node_id[_g]
                gene_total_split_reads[_n] = gene_total_split_reads.get(_n, 0.0) + split_reads_row
        for _g in acceptor_genes:
            if _g in gene_to_node_id:
                _n = gene_to_node_id[_g]
                gene_total_split_reads[_n] = gene_total_split_reads.get(_n, 0.0) + split_reads_row

        if not donor_genes or not acceptor_genes:
            continue

        for g1 in donor_genes:
            if g1 in gene_to_node_id:
                bucket = promiscuity.setdefault(g1, set())
                for g2 in acceptor_genes:
                    if g2 != g1:
                        bucket.add(g2)
        for g2 in acceptor_genes:
            if g2 in gene_to_node_id:
                bucket = promiscuity.setdefault(g2, set())
                for g1 in donor_genes:
                    if g1 != g2:
                        bucket.add(g1)

        for g1 in donor_genes:
            for g2 in acceptor_genes:
                if g1 not in gene_to_node_id or g2 not in gene_to_node_id:
                    continue
                n1 = gene_to_node_id[g1]
                n2 = gene_to_node_id[g2]
                _stats = gene_pair_stats.setdefault((n1, n2), {"num_junctions": 0, "max_split": 0.0, "canonical": 0})
                _stats["num_junctions"] += 1
                if split_reads_row > _stats["max_split"]:
                    _stats["max_split"] = split_reads_row
                if int(_jtype_col[_i]) in (1, 2, 3):
                    _stats["canonical"] += 1

    promiscuity_count = {g: len(partners) for g, partners in promiscuity.items()}

    edge_index_list = []
    edge_attr_list = []
    breakpoint_list = []

    for _, row in chimeric_df.iterrows():
        donor_genes = _genes_at(row["chr_donor"], row["brkpt_donor"])
        acceptor_genes = _genes_at(row["chr_acceptor"], row["brkpt_acceptor"])

        if not donor_genes or not acceptor_genes:
            continue

        for g1 in donor_genes:
            for g2 in acceptor_genes:
                if g1 not in gene_to_node_id or g2 not in gene_to_node_id:
                    continue

                n1 = gene_to_node_id[g1]
                n2 = gene_to_node_id[g2]

                discordant = _count_discordant_near(
                    row["chr_donor"],
                    row["brkpt_donor"],
                    row["chr_acceptor"],
                    row["brkpt_acceptor"],
                )
                split_reads = float(row["num_split_reads"])

                dist = abs(row["brkpt_donor"] - row["brkpt_acceptor"]) if row["chr_donor"] == row["chr_acceptor"] else 1e7

                tpm1 = tpm_lookup.get(g1, 0.0)
                tpm2 = tpm_lookup.get(g2, 0.0)

                donor_in_exon = _breakpoint_in_exon(gene_exon_map, g1, row["brkpt_donor"])
                acceptor_in_exon = _breakpoint_in_exon(gene_exon_map, g2, row["brkpt_acceptor"])
                donor_rel_pos = _relative_position(gene_coords, g1, row["brkpt_donor"])
                acceptor_rel_pos = _relative_position(gene_coords, g2, row["brkpt_acceptor"])

                ffpm = split_reads / ffpm_denom
                max_balanced_anchor = float(row.get("max_balanced_anchor", 0) or 0)
                prom_donor = float(promiscuity_count.get(g1, 0))
                prom_acceptor = float(promiscuity_count.get(g2, 0))
                read_through = _is_read_through(g1, g2)

                feat = np.concatenate(
                    [
                        [
                            np.log1p(split_reads),
                            np.log1p(discordant),
                            split_reads / (discordant + 1.0),
                            np.log1p(dist),
                            np.log1p((tpm1 + 1e-6) / (tpm2 + 1e-6)),
                            1.0 if row["chr_donor"] != row["chr_acceptor"] else 0.0,
                        ],
                        _one_hot_junction(row["junction_type"]),
                        _encode_strand(row["strand_donor"], row["strand_acceptor"]),
                        [
                            np.log1p(float(row.get("repeat_left_len", 0) or 0)),
                            np.log1p(float(row.get("repeat_right_len", 0) or 0)),
                        ],
                        [donor_in_exon, acceptor_in_exon, donor_rel_pos, acceptor_rel_pos],
                        [
                            np.log1p(ffpm * 1000.0),
                            np.log1p(max_balanced_anchor),
                            np.log1p(prom_donor),
                            np.log1p(prom_acceptor),
                            read_through,
                        ],
                    ]
                )

                if mitelman_db is not None:
                    name1 = gene_id_to_name_tmp.get(g1, "")
                    name2 = gene_id_to_name_tmp.get(g2, "")
                    recurrence = float(mitelman_db.known_pair_count(name1, name2))
                    feat = np.concatenate(
                        [
                            feat,
                            [1.0 if recurrence > 0 else 0.0, np.log1p(recurrence)],
                        ]
                    )

                _s = gene_pair_stats[(n1, n2)]
                _nj = _s["num_junctions"]
                _frac_can = _s["canonical"] / _nj if _nj > 0 else 0.0
                pair_feats = np.array(
                    [
                        np.log1p(_nj),
                        np.log1p(_s["max_split"]),
                        _frac_can,
                        np.log1p(gene_total_split_reads.get(n1, 0.0)),
                        np.log1p(gene_total_split_reads.get(n2, 0.0)),
                        _frac_can * donor_in_exon * acceptor_in_exon,
                    ],
                    dtype=np.float32,
                )

                edge_index_list.append([n1, n2])
                edge_attr_list.append(np.concatenate([feat, pair_feats]))

                breakpoint_list.append(
                    {
                        "chr_donor": row["chr_donor"],
                        "brkpt_donor": int(row["brkpt_donor"]),
                        "strand_donor": row["strand_donor"],
                        "chr_acceptor": row["chr_acceptor"],
                        "brkpt_acceptor": int(row["brkpt_acceptor"]),
                        "strand_acceptor": row["strand_acceptor"],
                        "split_reads": split_reads,
                        "pct_canonical": _s["canonical"] / _nj if _nj > 0 else 0.0,
                    }
                )

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.from_numpy(np.array(edge_attr_list, dtype=np.float32))
    else:
        num_edge_feat = 29 + (2 if mitelman_db is not None else 0)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, num_edge_feat), dtype=torch.float)

    print(f"[*] Edges: {edge_index.shape[1]} | Edge features: {edge_attr.shape[1]}")

    gene_id_to_name = gtf_df[["gene_id", "gene_name"]].drop_duplicates("gene_id").set_index("gene_id")["gene_name"]
    gene_name_to_node_id = {name: nid for gid, nid in gene_to_node_id.items() if (name := gene_id_to_name.get(gid)) is not None}

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if edge_attr.shape[0] > 0:
        graph.edge_evidence = edge_attr[:, 0].clone()
    else:
        graph.edge_evidence = torch.zeros((0,), dtype=torch.float32)
    graph.gene_to_node_id = gene_to_node_id
    graph.gene_name_to_node_id = gene_name_to_node_id
    graph.edge_breakpoints = breakpoint_list
    return graph
