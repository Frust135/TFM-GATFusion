import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import graph_builder
from data_parser.gtf_dictionary import load_gene_annotations, load_exon_annotations
from data_parser.load_chimeric import load_chimeric
from data_parser.load_discordant_pairs import load_discordant_pairs
from data_parser.load_mitelman import load_mitelman
from data_parser.load_reads_per_gene import load_expression
from model import build_model_from_checkpoint


def _predict_cache_key(bam_file, chimeric_file, reads_per_gene_file, gtf_file, library_type, min_split_reads, has_mitelman):
    """Hash for a graph cache key based on the input files and parameters."""
    from scripts.train_multi_graph import FEATURE_VERSION

    parts = []
    for fpath in (bam_file, chimeric_file, reads_per_gene_file, gtf_file):
        p = Path(fpath).resolve()
        parts.append(f"{p}|{p.stat().st_size}" if p.exists() else str(p))
    parts.append(f"lib={library_type}|msr={min_split_reads}|mit={has_mitelman}|fv={FEATURE_VERSION}")
    blob = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _build_or_load_predict_graph(
    bam_file,
    chimeric_file,
    reads_per_gene_file,
    gtf_file,
    library_type,
    min_split_reads,
    mitelman_db,
    cache_dir,
    bam_threads=1,
):
    """Build a graph, loading from cache when its available"""
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _predict_cache_key(
            bam_file,
            chimeric_file,
            reads_per_gene_file,
            gtf_file,
            library_type,
            min_split_reads,
            mitelman_db is not None,
        )
        cache_path = cache_dir / f"graph_{key}.pt"
        if cache_path.exists():
            graph = torch.load(cache_path, weights_only=False)
            print(f"[*] Graph loaded from cache ({graph.x.shape[0]} nodes | " f"{graph.edge_index.shape[1]} edges) -> {cache_path}")
            return graph

    disc = load_discordant_pairs(bam_file, threads=bam_threads)
    chim = load_chimeric(chimeric_file, min_split_reads=min_split_reads)
    expr = load_expression(reads_per_gene_file, library_type=library_type)
    gtf = load_gene_annotations(gtf_file)
    exon = load_exon_annotations(gtf_file)
    graph = graph_builder.create_graph(expr, chim, disc, gtf, mitelman_db=mitelman_db, exon_df=exon)

    if cache_path is not None:
        torch.save(graph, cache_path)
        print(f"[*] Graph cached -> {cache_path}")
    return graph


def _build_interval_index(df):
    """Build a per-chromosome interval from a GTF/exon DataFrame."""
    index = {}
    for chrom, group in df.groupby("chr"):
        group = group.sort_values("start")
        index[chrom] = (
            group["start"].to_numpy(),
            group["end"].to_numpy(),
            group["gene_id"].to_numpy(),
        )
    return index


def _annotate_breakpoints_batch(gene_index, exon_index, chroms, positions):
    """Annotate a list of breakpoints in one vectorised pass"""
    E = len(chroms)
    out = ["intergenic"] * E
    if E == 0:
        return out

    chroms_arr = np.asarray(chroms, dtype=object)
    pos_arr = np.asarray(positions, dtype=np.int64)

    by_chrom = {}
    for i, c in enumerate(chroms_arr):
        by_chrom.setdefault(c, []).append(i)

    for chrom, idx_list in by_chrom.items():
        if chrom not in gene_index:
            continue
        idx = np.asarray(idx_list, dtype=np.int64)
        ps = pos_arr[idx]

        g_starts, g_ends, g_ids = gene_index[chrom]
        hi = np.searchsorted(g_starts, ps, side="right")
        overlap_sets = [None] * len(idx)
        any_overlap = np.zeros(len(idx), dtype=bool)

        e_entry = exon_index.get(chrom)
        if e_entry is not None:
            e_starts, e_ends, e_gene_ids = e_entry
            hi_e = np.searchsorted(e_starts, ps, side="right")
        else:
            e_starts = e_ends = e_gene_ids = None
            hi_e = np.zeros(len(idx), dtype=np.int64)

        for k in range(len(idx)):
            h = int(hi[k])
            if h == 0:
                continue
            p = int(ps[k])
            mask = g_ends[:h] >= p
            if not mask.any():
                continue
            any_overlap[k] = True
            gset = g_ids[:h][mask]
            overlap_sets[k] = gset

            if e_starts is None:
                continue
            he = int(hi_e[k])
            if he == 0:
                continue
            e_mask = e_ends[:he] >= p
            if not e_mask.any():
                continue
            cand_ids = e_gene_ids[:he][e_mask]
            if np.isin(cand_ids, gset).any():
                out[int(idx[k])] = "exon"
            else:
                out[int(idx[k])] = "intron"
        for k in np.flatnonzero(any_overlap):
            i_full = int(idx[k])
            if out[i_full] == "intergenic":
                out[i_full] = "intron"

    return out


def predict(args):
    print("\n=== Step 1: Loading input data ===")
    gtf_data = load_gene_annotations(args.gtf_file)
    exon_data = load_exon_annotations(args.gtf_file)

    mitelman_db = None
    if args.mitelman_file:
        mitelman_db = load_mitelman(args.mitelman_file)

    print("\n=== Step 2: Building graph ===")
    cache_dir = args.graph_cache_dir if (args.graph_cache_dir and not args.no_cache) else None
    graph = _build_or_load_predict_graph(
        args.bam_file,
        args.chimeric_file,
        args.reads_per_gene_file,
        args.gtf_file,
        args.library_type,
        args.min_split_reads,
        mitelman_db,
        cache_dir,
        bam_threads=args.bam_threads,
    )
    if graph.edge_index.shape[1] == 0:
        print("[!] No edges built from the input data. Nothing to predict.")
        sys.exit(1)

    print("\n=== Step 3: Loading model ===")
    model, checkpoint, _ = build_model_from_checkpoint(
        args.model,
        fallback_config={
            "num_node_features": graph.x.shape[1],
            "num_edge_features": graph.edge_attr.shape[1],
            "hidden_dim": args.hidden_dim,
            "edge_embed_dim": args.edge_embed_dim,
            "dropout": 0.3,
            "prior_pos_rate": None,
        },
    )
    model.eval()
    print(f"[*] Model loaded from {args.model}")

    metadata = checkpoint.get("metadata", {}) or {}
    edge_norm_stats = metadata.get("edge_norm_stats")
    if edge_norm_stats is not None and graph.edge_attr.shape[0] > 0:
        from scripts.train_multi_graph import apply_edge_norm

        apply_edge_norm([graph], edge_norm_stats)
        print(f"[*] Applied global edge normalization ({len(edge_norm_stats['continuous_cols'])} cols).")
    else:
        print("[!] No edge_norm_stats in checkpoint — edges will use raw feature values.")

    metadata = checkpoint.get("metadata", {}) or {}
    if args.threshold is None:
        ckpt_thr = metadata.get("best_val_thr")
        if ckpt_thr is not None:
            args.threshold = float(ckpt_thr)
            print(f"[*] Using checkpoint's best validation threshold: {args.threshold:.4f}")
        else:
            args.threshold = 0.5
            print("[*] No best_val_thr found in checkpoint metadata — falling back to 0.5.")

    print("\n=== Step 4: Running prediction ===")
    with torch.no_grad():
        logits = model(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            edge_evidence=getattr(graph, "edge_evidence", None),
        )

    platt_params = metadata.get("platt_params")
    if platt_params is not None:
        logits = model.calibrate_logits(logits, platt_params)
        print(f"[*] Applied Platt scaling (A={platt_params['A']:.4f}, B={platt_params['B']:.4f}).")

    scores = torch.sigmoid(logits).reshape(-1)

    print("[*] Building breakpoint annotation index...")
    gene_index = _build_interval_index(gtf_data)
    exon_index = _build_interval_index(exon_data)

    id_to_gene_id = {v: k for k, v in graph.gene_to_node_id.items()}
    _name_map = gtf_data.drop_duplicates("gene_id").set_index("gene_id")["gene_name"].to_dict()
    id_to_gene_name = {node_id: _name_map.get(gid) for gid, node_id in graph.gene_to_node_id.items() if _name_map.get(gid)}

    E = graph.edge_index.shape[1]
    breakpoints = [graph.edge_breakpoints[i] for i in range(E)]
    donor_chroms = [bp["chr_donor"] for bp in breakpoints]
    accept_chroms = [bp["chr_acceptor"] for bp in breakpoints]

    donor_pos = [(bp["brkpt_donor"] - 1) if bp["strand_donor"] == "+" else (bp["brkpt_donor"] + 1) for bp in breakpoints]
    accept_pos = [(bp["brkpt_acceptor"] + 1) if bp["strand_acceptor"] == "+" else (bp["brkpt_acceptor"] - 1) for bp in breakpoints]

    print(f"[*] Annotating {len(breakpoints)} breakpoints...")
    donor_annots = _annotate_breakpoints_batch(gene_index, exon_index, donor_chroms, donor_pos)
    accept_annots = _annotate_breakpoints_batch(gene_index, exon_index, accept_chroms, accept_pos)

    print("[*] Building result table...")
    ei_np = graph.edge_index.cpu().numpy()
    src_ids = ei_np[0]
    dst_ids = ei_np[1]
    donor_names = [id_to_gene_name.get(int(s), id_to_gene_id.get(int(s), str(int(s)))) for s in src_ids]
    acceptor_names = [id_to_gene_name.get(int(d), id_to_gene_id.get(int(d), str(int(d)))) for d in dst_ids]
    scores_np = scores.cpu().numpy()

    df_all = pd.DataFrame(
        {
            "donor_gene": donor_names,
            "acceptor_gene": acceptor_names,
            "score": np.round(scores_np, 4),
            "split_reads": [int(bp["split_reads"]) for bp in breakpoints],
            "chr_donor": donor_chroms,
            "brkpt_donor": donor_pos,
            "strand_donor": [bp["strand_donor"] for bp in breakpoints],
            "donor_region": donor_annots,
            "chr_acceptor": accept_chroms,
            "brkpt_acceptor": accept_pos,
            "strand_acceptor": [bp["strand_acceptor"] for bp in breakpoints],
            "acceptor_region": accept_annots,
            "pct_canonical": [round(bp.get("pct_canonical", 0.0), 3) for bp in breakpoints],
        }
    )
    same_gene = df_all["donor_gene"].values == df_all["acceptor_gene"].values
    same_chr = df_all["chr_donor"].values == df_all["chr_acceptor"].values
    ftype = np.where(same_gene, "intra-genic", np.where(same_chr, "intra-chromosomal", "inter-chromosomal"))
    df_all["fusion_type"] = ftype

    minus_mask = df_all["strand_donor"] == "-"
    if minus_mask.any():
        swap_pairs = [
            ("donor_gene", "acceptor_gene"),
            ("chr_donor", "chr_acceptor"),
            ("brkpt_donor", "brkpt_acceptor"),
            ("strand_donor", "strand_acceptor"),
            ("donor_region", "acceptor_region"),
        ]
        for col_a, col_b in swap_pairs:
            if col_a in df_all.columns and col_b in df_all.columns:
                tmp = df_all.loc[minus_mask, col_a].values.copy()
                df_all.loc[minus_mask, col_a] = df_all.loc[minus_mask, col_b].values
                df_all.loc[minus_mask, col_b] = tmp

    dedup_key = ["chr_donor", "brkpt_donor", "chr_acceptor", "brkpt_acceptor"]
    if len(df_all) > 1:
        df_all = df_all.sort_values("score", ascending=False)
        sr_sum = df_all.groupby(dedup_key, sort=False)["split_reads"].sum().rename("split_reads_sum")
        df_all = df_all.drop_duplicates(subset=dedup_key, keep="first").copy()
        df_all = df_all.join(sr_sum, on=dedup_key)
        df_all["split_reads"] = df_all["split_reads_sum"].astype(int)
        df_all = df_all.drop(columns=["split_reads_sum"])

    df_all = df_all.sort_values("score", ascending=False).reset_index(drop=True)

    df_dropped = pd.DataFrame()
    if not args.no_postprocess and not df_all.empty:
        from utils.postprocess import apply_postprocess

        df_all, df_dropped = apply_postprocess(
            df_all,
            drop_intragenic=not args.keep_intragenic,
            drop_readthrough=not args.keep_readthrough,
            drop_blacklisted=not args.keep_blacklisted,
            drop_noncanonical_artifacts=not args.keep_noncanonical,
            noncanonical_min_pct=args.noncanonical_min_pct,
            readthrough_max_dist=args.readthrough_max_dist,
            annotate_only=args.postprocess_annotate_only,
        )

    print(f"[*] Candidate junctions evaluated: {len(df_all)}")
    if args.top_k is not None:
        df_predicted = df_all.head(args.top_k).copy()
        min_score = df_predicted["score"].min() if not df_predicted.empty else 0.0
        print(f"[*] Selection: top-{args.top_k} edges (lowest score in set: {min_score:.4f})")
    else:
        df_predicted = df_all[df_all["score"] >= args.threshold].copy()
        print(f"[*] Selection: score >= {args.threshold:.4f} — " f"{len(df_predicted)} fusion(s) reported")

    if not df_predicted.empty:
        print("\n--- Predicted Fusions ---")
        print(df_predicted.to_string(index=False))
    else:
        print("\n[!] No fusions found. Try lowering --threshold or using --top_k 10.")

    if args.output:
        df_predicted.to_csv(args.output, index=False, sep="\t")
        print(f"[*] Results saved to {args.output}")

    return df_predicted


def main():
    parser = argparse.ArgumentParser(
        description="Predict gene fusions using a trained FusionPredictor checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bam_file", required=True, help="Sorted BAM file")
    parser.add_argument("--chimeric_file", required=True, help="STAR Chimeric.out.junction file")
    parser.add_argument("--reads_per_gene_file", required=True, help="STAR ReadsPerGene.out.tab file")
    parser.add_argument("--gtf_file", required=True, help="Reference annotation GTF")
    parser.add_argument("--model", required=True, help="Trained checkpoint (.pt file)")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Fallback hidden_dim if the checkpoint has no config")
    parser.add_argument("--edge_embed_dim", type=int, default=32, help="Fallback edge_embed_dim if the checkpoint has no config")
    parser.add_argument("--threshold", type=float, default=None, help="Minimum score to report a fusion. If omitted, uses the " "best_val_thr saved in the checkpoint (or 0.5 if missing).")
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Report the top-K highest-scoring edges regardless of their "
        "absolute score. Takes precedence over --threshold when both "
        "are given. Useful when model scores are compressed and a fixed "
        "threshold is unreliable (e.g. all scores < 0.5).",
    )
    parser.add_argument("--min_split_reads", type=int, default=2, help="Minimum split reads per chimeric junction")
    parser.add_argument("--library_type", default="unstranded", choices=["unstranded", "stranded_forward", "stranded_reverse"])
    parser.add_argument("--output", default=None, help="TSV file to save predictions (optional)")
    parser.add_argument("--graph_cache_dir", default="cache/graphs", help="Directory to cache built graphs. Empty string to disable.")
    parser.add_argument("--no_cache", action="store_true", help="Rebuild graph from scratch, ignoring cache.")
    parser.add_argument(
        "--mitelman_file",
        default=None,
        help="Optional Mitelman DB flat file. Must be the same file used during training " "(adds 2 node + 2 edge features; checkpoint must match).",
    )

    perf = parser.add_argument_group("performance")
    perf.add_argument(
        "--bam_threads",
        type=int,
        default=4,
        help="Decompression threads for BAM reading. BGZF decode is " "the dominant cost of the discordant-pair scan; 4-8 gives " "a near-linear speed-up on multi-core machines.",
    )
    perf.add_argument("--torch_threads", type=int, default=None, help="Sets torch.set_num_threads for the inference forward pass. " "Defaults to PyTorch's auto value when omitted.")

    pg = parser.add_argument_group("post-processing")
    pg.add_argument("--no_postprocess", action="store_true", help="Skip all post-prediction filters and functional annotation.")
    pg.add_argument("--postprocess_annotate_only", action="store_true", help="Annotate functional class and filter reasons but keep ALL rows.")
    pg.add_argument("--keep_intragenic", action="store_true", help="Do not drop edges where donor == acceptor gene.")
    pg.add_argument("--keep_readthrough", action="store_true", help="Do not drop same-chromosome adjacent same-strand events.")
    pg.add_argument("--keep_blacklisted", action="store_true", help="Do not drop edges touching ribosomal/mitochondrial/etc. artefact families.")
    pg.add_argument("--readthrough_max_dist", type=int, default=100_000, help="Max breakpoint distance (bp) to flag as read-through.")
    pg.add_argument(
        "--keep_noncanonical",
        action="store_true",
        help="Do not drop inter-chromosomal exon-exon junctions with no " "canonical splice-site signal (pct_canonical < threshold). " "By default these are removed as sequence-homology artefacts.",
    )
    pg.add_argument(
        "--noncanonical_min_pct",
        type=float,
        default=0.1,
        help="Minimum fraction of reads with canonical junction type (1 or 2) " "required to keep an inter-chromosomal exon-exon edge. " "Edges below this are classified as non-canonical artefacts.",
    )

    args = parser.parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
        print(f"[*] PyTorch intra-op threads set to: {args.torch_threads}")
    predict(args)


if __name__ == "__main__":
    main()
