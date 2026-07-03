import argparse
import copy
import csv
import hashlib
import io
import math
import multiprocessing
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
FEATURE_VERSION = "v8"  # validate the cache version

import graph_builder
from data_parser.gtf_dictionary import load_gene_annotations, load_exon_annotations
from data_parser.load_chimeric import load_chimeric
from data_parser.load_discordant_pairs import load_discordant_pairs
from data_parser.load_mitelman import load_mitelman
from data_parser.load_reads_per_gene import load_expression
from model import FusionPredictor, build_model_from_checkpoint, create_checkpoint, _f1_best_threshold, _f1_at_threshold
from utils.create_train_data import create_train_data


def _get_continuous_edge_cols(num_edge_features):
    """Indices of continuous (non one-hot) edge features, for z-score normalization."""
    cols = [0, 1, 2, 3, 4, 13, 14, 17, 18, 19, 20, 21, 22]
    if num_edge_features == 31:  # with Mitelman recurrence features
        cols += [25, 26, 27, 28, 29, 30]
    else:
        cols += [24, 25, 26, 27, 28]
    return cols


def compute_global_edge_norm(graphs, continuous_cols):
    """Mean/std per continuous edge column, computed over all edges in the graph."""
    all_edges = torch.cat([g.edge_attr for g in graphs if g.edge_attr.shape[0] > 0], dim=0)
    mean_vals, std_vals = [], []
    for col in continuous_cols:
        col_data = all_edges[:, col]
        mean_vals.append(float(col_data.mean()))
        std_vals.append(float(col_data.std()) + 1e-6)
    print(f"[*] Global edge normalization computed over {all_edges.shape[0]:,} edges, {len(continuous_cols)} continuous columns.")
    return {"continuous_cols": continuous_cols, "mean": mean_vals, "std": std_vals}


def apply_edge_norm(graphs, norm_stats):
    """Apply pre-computed z-score normalization to a list of graphs, in place."""
    cols = norm_stats["continuous_cols"]
    means = norm_stats["mean"]
    stds = norm_stats["std"]
    for g in graphs:
        if g.edge_attr.shape[0] == 0:
            continue
        for i, col in enumerate(cols):
            g.edge_attr[:, col] = (g.edge_attr[:, col] - means[i]) / stds[i]


def _deep_copy_graphs(graphs):
    """Independent copies the graphs"""
    return [copy.deepcopy(g) for g in graphs]


# --- Cache ---


def _graph_worker(args_tuple):
    """Worker function for building a graph in a separate process."""
    row, default_gtf, library_type, min_split_reads, mitelman_db, cache_dir = args_tuple
    buf = io.StringIO()
    with redirect_stdout(buf):
        graph = build_or_load_graph(row, default_gtf, library_type, min_split_reads, mitelman_db, cache_dir)
    return graph, buf.getvalue()


def _cache_key(row, default_gtf, library_type, min_split_reads, has_mitelman):
    """Hash of a manifest row + build params. Any change of the parameters will invalidate the cache."""
    parts = []
    gtf_file = row.get("gtf_file") or default_gtf or ""
    for field in ("bam_file", "chimeric_file", "reads_per_gene_file"):
        fpath = row.get(field, "")
        resolved = str(Path(fpath).resolve()) if fpath else ""
        if fpath and Path(fpath).exists():
            parts.append(f"{resolved}|{Path(fpath).stat().st_size}")
        else:
            parts.append(resolved)
    if gtf_file and Path(gtf_file).exists():
        resolved = str(Path(gtf_file).resolve())
        parts.append(f"{resolved}|{Path(gtf_file).stat().st_size}")
    else:
        parts.append(str(Path(gtf_file).resolve()) if gtf_file else "")

    fusions_raw = row.get("positive_fusions", "")
    fusions_file = row.get("positive_fusions_file", "")
    if fusions_file and Path(fusions_file).exists():
        resolved = str(Path(fusions_file).resolve())
        parts.append(f"{resolved}|{Path(fusions_file).stat().st_size}")
    else:
        parts.append(fusions_raw)

    parts.append(f"lib={library_type}|msr={min_split_reads}|mit={has_mitelman}|fv={FEATURE_VERSION}")
    blob = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def build_or_load_graph(row, default_gtf, library_type, min_split_reads, mitelman_db=None, cache_dir=None):
    """Load a cached graph if available, otherwise build and cache it."""
    cache_path = None
    if cache_dir is not None:
        key = _cache_key(row, default_gtf, library_type, min_split_reads, mitelman_db is not None)
        cache_path = Path(cache_dir) / f"graph_{key}.pt"
        if cache_path.exists():
            graph = torch.load(cache_path, weights_only=False)
            positives = int(graph.y.sum().item())
            print(f"\n{'=' * 50}")
            print(f"[*] Row {row['__row__']}: loaded from cache ({graph.x.shape[0]} nodes | {graph.edge_index.shape[1]} edges | {positives} fusions)")
            return graph

    graph = build_graph_for_row(row, default_gtf, library_type, min_split_reads, mitelman_db)

    if graph is not None and cache_path is not None:
        torch.save(graph, cache_path)
        print(f"  [*] Cached -> {cache_path}")

    return graph


# --- Training TSV loading ---


def _detect_delimiter(file_path):
    with open(file_path, "r", encoding="utf-8") as fh:
        sample = fh.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        return dialect.delimiter
    except csv.Error:
        return "\t"


def load_manifest(file_path):
    delimiter = _detect_delimiter(file_path)
    with open(file_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        rows = []
        for index, row in enumerate(reader, start=1):
            normalized = {k.strip(): (v or "").strip() for k, v in row.items() if k}
            if not any(normalized.values()):
                continue
            normalized["__row__"] = index
            rows.append(normalized)
    if not rows:
        raise ValueError(f"Manifest is empty: {file_path}")
    return rows


# --- Fusion label parsing ---


def _parse_fusion_pair(raw):
    """Parse 'GENE1:GENE2' or 'GENE1:GENE2@chrA:posA-chrB:posB'."""
    if "@" in raw:
        gene_part, brkpt_part = raw.split("@", 1)
    else:
        gene_part = raw
        brkpt_part = None

    for sep in (":", ",", "\t"):
        if sep in gene_part:
            donor, acceptor = gene_part.split(sep, 1)
            donor = donor.strip()
            acceptor = acceptor.strip()
            if brkpt_part:
                try:
                    left, right = brkpt_part.split("-", 1)
                    chr_a, pos_a = left.rsplit(":", 1)
                    chr_b, pos_b = right.rsplit(":", 1)
                    return (donor, acceptor, chr_a.strip(), int(pos_a), chr_b.strip(), int(pos_b))
                except (ValueError, TypeError):
                    return donor, acceptor
            return donor, acceptor
    raise ValueError(f"Cannot parse fusion '{raw}'. Expected DONOR:ACCEPTOR format.")


def _collect_positive_fusions(row):
    raw = row.get("positive_fusions", "")
    if raw:
        return [_parse_fusion_pair(item) for item in raw.split(";") if item.strip()]

    fusions_file = row.get("positive_fusions_file", "")
    if fusions_file:
        pairs = []
        with open(fusions_file, "r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    pairs.append(_parse_fusion_pair(stripped))
                except ValueError as exc:
                    raise ValueError(f"{exc} (file: {fusions_file}, line {line_num})") from exc
        return pairs

    raise ValueError(f"Row {row['__row__']}: provide 'positive_fusions' or 'positive_fusions_file'.")


def build_graph_for_row(row, default_gtf, library_type, min_split_reads, mitelman_db=None):
    """Graph construction for a single sample row"""
    gtf_file = row.get("gtf_file") or default_gtf
    if not gtf_file:
        raise ValueError(f"Row {row['__row__']}: no gtf_file provided and no global --gtf_file set.")

    for field in ("bam_file", "chimeric_file", "reads_per_gene_file"):
        path = row.get(field, "")
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"Row {row['__row__']}: {field} not found: '{path}'")
    if not Path(gtf_file).exists():
        raise FileNotFoundError(f"Row {row['__row__']}: gtf_file not found: '{gtf_file}'")

    print(f"\n{'=' * 50}")
    print(f"[*] Row {row['__row__']}: building graph...")

    discordant = load_discordant_pairs(row["bam_file"])
    chimeric = load_chimeric(row["chimeric_file"], min_split_reads=min_split_reads)
    reads_per_gene = load_expression(row["reads_per_gene_file"], library_type=library_type)
    gtf = load_gene_annotations(gtf_file)
    exon_df = load_exon_annotations(gtf_file)

    graph = graph_builder.create_graph(
        reads_per_gene,
        chimeric,
        discordant,
        gtf,
        mitelman_db=mitelman_db,
        exon_df=exon_df,
    )

    if graph.edge_index.shape[1] == 0:
        print(f"  [!] Row {row['__row__']}: no edges found — sample will be skipped.")
        return None

    positive_fusions = _collect_positive_fusions(row)
    graph.y = create_train_data(
        graph.edge_index,
        graph.gene_to_node_id,
        positive_fusions,
        gene_name_to_node_id=graph.gene_name_to_node_id,
        edge_breakpoints=graph.edge_breakpoints,
    )

    del graph.gene_to_node_id
    del graph.gene_name_to_node_id
    del graph.edge_breakpoints

    positives = int(graph.y.sum().item())
    print(f"  [+] Graph ready: {graph.x.shape[0]} nodes | {graph.edge_index.shape[1]} edges | {positives} labeled fusions")

    return graph


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train the fusion GNN on multiple graphs simultaneously (mini-batch training).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="TSV/CSV manifest with one sample per row.")
    parser.add_argument("--gtf_file", default=None, help="Default GTF file for rows that omit the gtf_file column.")
    parser.add_argument("--output_model", default="checkpoints/multi_graph.pt", help="Path where the final checkpoint will be saved.")
    parser.add_argument("--resume_from", default=None, help="Optional existing checkpoint to fine-tune from.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--mitelman_file", default=None, help="Optional Mitelman DB flat file (TSV/CSV with gene_a / gene_b columns).")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--edge_embed_dim", type=int, default=32, help="Dimension of the learned edge embedding produced by the edge encoder MLP.")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--heads", type=int, default=2, help="GATv2 attention heads per layer.")
    parser.add_argument("--num_gnn_layers", type=int, default=2, help="Number of GATv2 layers.")
    parser.add_argument("--no_node_skip", action="store_false", dest="node_skip", help="Disable the raw-node skip pathway into the classifier.")
    parser.set_defaults(node_skip=True)
    parser.add_argument("--batch_size", type=int, default=4, help="Number of graphs per mini-batch.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience in epochs.")
    parser.add_argument("--neg_ratio", type=int, default=50, help="Max negatives per positive for subsampling (0 = no subsampling). Ignored when focal loss is enabled.")
    parser.add_argument("--no_focal_loss", action="store_false", dest="focal_loss", help="Disable focal loss and use BCE+pos_weight instead (focal loss is on by default).")
    parser.set_defaults(focal_loss=True)
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma parameter for focal loss (higher = more focus on hard examples).")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of graphs held out for validation (0 to disable). Ignored when --cv_folds is set.")
    parser.add_argument(
        "--cv_folds", type=int, default=0, help="Number of stratified CV folds. When > 1, trains K models then retrains a final model on the full dataset. 0 = use --val_split instead."
    )
    parser.add_argument("--select_metric", default="auprc", choices=["auprc", "f1"], help="Primary metric for model selection during training.")
    parser.add_argument("--feature_noise_std", type=float, default=0.01, help="Std of Gaussian noise added to continuous edge features of positive edges during training. 0 to disable.")
    parser.add_argument("--drop_empty_graphs", action="store_true", help="Drop training graphs with zero positive labels.")
    parser.add_argument("--graph_cache_dir", default="cache/graphs", help="Directory to cache built graphs. Empty string disables caching.")
    parser.add_argument("--no_cache", action="store_true", help="Rebuild all graphs from scratch, ignoring cached files.")
    parser.add_argument("--min_split_reads", type=int, default=2, help="Minimum split reads to include a chimeric junction.")
    parser.add_argument("--library_type", default="unstranded", choices=["unstranded", "stranded_forward", "stranded_reverse"])
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help=f"Number of parallel processes for building graphs from raw files (0 or 1 = sequential). This machine has {os.cpu_count()} logical cores.",
    )
    parser.add_argument(
        "--dataloader_workers",
        type=int,
        default=0,
        help="Number of PyTorch Geometric DataLoader worker processes that prefetch mini-batches (0 = load in the main process).",
    )
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=None,
        help=f"CPU threads PyTorch uses for intra-op parallelism (matmuls, convolutions). Defaults to all available cores ({os.cpu_count()}).",
    )
    return parser.parse_args()


def _configure_torch_threads(args):
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
        print(f"[*] PyTorch intra-op threads set to: {args.torch_threads}")
    else:
        print(f"[*] PyTorch intra-op threads: {torch.get_num_threads()} (auto)")


def _resolve_cache_dir(args):
    if not args.graph_cache_dir or args.no_cache:
        print("[*] Graph cache: disabled")
        return None
    cache_dir = Path(args.graph_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Graph cache: {cache_dir}")
    return cache_dir


def _build_dataset(rows, args, mitelman_db, cache_dir):
    """Build one graph per TSV row, being in parallel if num_workers > 1."""
    dataset = []
    num_workers = max(1, args.num_workers)
    if num_workers > 1:
        print(f"[*] Building graphs in parallel with {num_workers} workers...")
        worker_args = [(row, args.gtf_file, args.library_type, args.min_split_reads, mitelman_db, cache_dir) for row in rows]
        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as executor:
            futures = [executor.submit(_graph_worker, arg) for arg in worker_args]
            for future in as_completed(futures):
                graph, output = future.result()
                sys.stdout.write(output)
                if graph is not None:
                    dataset.append(graph)
    else:
        print("[*] Building graphs sequentially (num_workers=1)...")
        for row in rows:
            graph = build_or_load_graph(row, args.gtf_file, args.library_type, args.min_split_reads, mitelman_db, cache_dir)
            if graph is not None:
                dataset.append(graph)
    return dataset


def _build_model_config(args, num_node_features, num_edge_features, prior_pos_rate):
    return {
        "num_node_features": num_node_features,
        "num_edge_features": num_edge_features,
        "hidden_dim": args.hidden_dim,
        "edge_embed_dim": args.edge_embed_dim,
        "dropout": args.dropout,
        "heads": args.heads,
        "num_gnn_layers": args.num_gnn_layers,
        "node_skip": args.node_skip,
        "prior_pos_rate": prior_pos_rate,
    }


def _run_cv_folds(dataset, args, manifest_path, output_path):
    """K-fold stratified CV: trains K models to get an aggregated val threshold/AUPRC, then retrains on all data."""
    K = args.cv_folds
    rng = random.Random(42)
    pos_graphs = [i for i, g in enumerate(dataset) if g.y.sum().item() > 0]
    neg_graphs = [i for i, g in enumerate(dataset) if g.y.sum().item() == 0]
    rng.shuffle(pos_graphs)
    rng.shuffle(neg_graphs)

    folds = [[] for _ in range(K)]
    for j, idx in enumerate(pos_graphs):
        folds[j % K].append(idx)
    for j, idx in enumerate(neg_graphs):
        folds[j % K].append(idx)

    print(f"\n{'=' * 50}")
    print(f"[*] {K}-fold stratified cross-validation")
    print(f"[*] Fold sizes: {[len(f) for f in folds]} (total {len(dataset)})")

    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    for i, g in enumerate(dataset[1:], start=2):
        if g.x.shape[1] != num_node_features:
            raise ValueError(f"Graph {i} has {g.x.shape[1]} node features, expected {num_node_features}.")
        if g.edge_attr.shape[1] != num_edge_features:
            raise ValueError(f"Graph {i} has {g.edge_attr.shape[1]} edge features, expected {num_edge_features}.")

    continuous_cols = _get_continuous_edge_cols(num_edge_features)

    fold_metrics = []
    all_val_logits = []
    all_val_targets = []

    for fold_i in range(K):
        print(f"\n{'=' * 50}")
        print(f"[*] FOLD {fold_i + 1}/{K}")

        val_idx = set(folds[fold_i])
        fold_train = _deep_copy_graphs([dataset[i] for i in range(len(dataset)) if i not in val_idx])
        fold_val = _deep_copy_graphs([dataset[i] for i in sorted(val_idx)])

        if args.drop_empty_graphs:
            fold_train = [g for g in fold_train if g.y.sum().item() > 0]

        fold_norm = compute_global_edge_norm(fold_train, continuous_cols)
        apply_edge_norm(fold_train, fold_norm)
        apply_edge_norm(fold_val, fold_norm)

        fold_train_pos = sum(int(g.y.sum().item()) for g in fold_train)
        fold_train_neg = sum(int((g.y == 0).sum().item()) for g in fold_train)
        prior = fold_train_pos / max(fold_train_pos + fold_train_neg, 1)

        fold_config = _build_model_config(args, num_node_features, num_edge_features, prior)
        fold_model = FusionPredictor(**fold_config)

        fold_summary = fold_model.train_model_multi_graph(
            fold_train,
            val_dataset=fold_val,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            batch_size=args.batch_size,
            dataloader_workers=args.dataloader_workers,
            neg_ratio=args.neg_ratio if args.neg_ratio > 0 else None,
            use_focal_loss=args.focal_loss,
            focal_gamma=args.focal_gamma,
            feature_noise_std=args.feature_noise_std,
            select_metric=args.select_metric,
        )
        fold_metrics.append(fold_summary)

        fold_model.eval()
        device = next(fold_model.parameters()).device
        with torch.no_grad():
            for g in fold_val:
                g = g.to(device)
                logits = fold_model(g.x, g.edge_index, g.edge_attr, edge_evidence=getattr(g, "edge_evidence", None))
                all_val_logits.append(logits.cpu())
                all_val_targets.append(g.y.cpu())

    cv_val_logits = torch.cat(all_val_logits)
    cv_val_targets = torch.cat(all_val_targets)
    cv_probs = torch.sigmoid(cv_val_logits)
    _, cv_best_thr, _, _, cv_auprc = _f1_best_threshold(cv_probs, cv_val_targets)
    cv_f1, _, _ = _f1_at_threshold(cv_probs, cv_val_targets, cv_best_thr)
    avg_fold_auprc = sum(s.get("best_val_auprc", 0) for s in fold_metrics) / K

    print(f"\n{'=' * 50}")
    print(f"[*] CV RESULTS ({K} folds):")
    print(f"    Aggregated AUPRC:      {cv_auprc:.4f}")
    print(f"    Mean per-fold AUPRC:   {avg_fold_auprc:.4f}")
    print(f"    Aggregated best F1:    {cv_f1:.4f} @ threshold={cv_best_thr:.4f}")
    print(f"    Total val edges:       {cv_val_targets.numel():,} ({int(cv_val_targets.sum().item())} positives)")

    print(f"\n{'=' * 50}")
    print(f"[*] Retraining final model on ALL {len(dataset)} graphs...")

    final_dataset = _deep_copy_graphs(dataset)
    if args.drop_empty_graphs:
        final_dataset = [g for g in final_dataset if g.y.sum().item() > 0]

    edge_norm_stats = compute_global_edge_norm(final_dataset, continuous_cols)
    apply_edge_norm(final_dataset, edge_norm_stats)

    train_pos = sum(int(g.y.sum().item()) for g in final_dataset)
    train_neg = sum(int((g.y == 0).sum().item()) for g in final_dataset)
    prior_pos_rate = train_pos / max(train_pos + train_neg, 1)

    model_config = _build_model_config(args, num_node_features, num_edge_features, prior_pos_rate)
    model = FusionPredictor(**model_config)
    summary = model.train_model_multi_graph(
        final_dataset,
        val_dataset=None,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience * 2,
        batch_size=args.batch_size,
        dataloader_workers=args.dataloader_workers,
        neg_ratio=args.neg_ratio if args.neg_ratio > 0 else None,
        use_focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        feature_noise_std=args.feature_noise_std,
        select_metric=args.select_metric,
    )

    platt_params = model.fit_platt_scaling(final_dataset)

    checkpoint = create_checkpoint(
        model,
        config=model_config,
        optimizer_state_dict=summary["optimizer_state_dict"],
        extra_metadata={
            "epochs_ran": summary["epochs_ran"],
            "best_loss": summary["best_loss"],
            "best_val_f1": cv_f1,
            "best_train_f1": summary.get("best_train_f1", -1.0),
            "best_val_thr": cv_best_thr,
            "best_val_auprc": cv_auprc,
            "cv_folds": K,
            "cv_mean_auprc": avg_fold_auprc,
            "num_graphs": len(dataset),
            "num_train_graphs": len(final_dataset),
            "num_val_graphs": 0,
            "manifest": str(manifest_path),
            "edge_norm_stats": edge_norm_stats,
            "platt_params": platt_params,
        },
    )
    torch.save(checkpoint, output_path)
    print(f"\n[*] Checkpoint saved to {output_path}")
    print(f"[*] Recommended inference threshold (from {K}-fold CV): {cv_best_thr:.4f}")
    print(f"[*] CV AUPRC: {cv_auprc:.4f}")


def _run_single_split(dataset, args, manifest_path, output_path):
    """Single train/val split."""
    if args.val_split > 0 and len(dataset) > 1:
        n_val_total = max(1, int(len(dataset) * args.val_split))
        rng = random.Random(42)
        pos_graphs = [i for i, g in enumerate(dataset) if g.y.sum().item() > 0]
        neg_graphs = [i for i, g in enumerate(dataset) if g.y.sum().item() == 0]
        rng.shuffle(pos_graphs)
        rng.shuffle(neg_graphs)

        n_val_pos = max(1, int(round(len(pos_graphs) * args.val_split))) if pos_graphs else 0
        n_val_neg = max(0, n_val_total - n_val_pos) if neg_graphs else 0
        n_val_neg = min(n_val_neg, len(neg_graphs))

        val_idx = set(pos_graphs[:n_val_pos]) | set(neg_graphs[:n_val_neg])
        val_dataset = [dataset[i] for i in sorted(val_idx)]
        train_dataset = [dataset[i] for i in range(len(dataset)) if i not in val_idx]
        print(f"[*] Stratified split — train graphs: {len(train_dataset)} | val graphs: {len(val_dataset)} (val: {n_val_pos} with positives + {n_val_neg} negative-only)")
    else:
        train_dataset = dataset
        val_dataset = None
        if args.val_split > 0:
            print("[!] Only 1 graph available — validation split skipped.")

    if args.drop_empty_graphs:
        before = len(train_dataset)
        train_dataset = [g for g in train_dataset if g.y.sum().item() > 0]
        dropped = before - len(train_dataset)
        if dropped > 0:
            print(f"[*] Dropped {dropped} train graphs with zero positives ({len(train_dataset)} remaining).")

    train_pos = sum(int(g.y.sum().item()) for g in train_dataset)
    train_neg = sum(int((g.y == 0).sum().item()) for g in train_dataset)
    print(f"[*] Train positives: {train_pos} | Train negatives: {train_neg:,} (ratio {train_neg / max(train_pos, 1):.0f}:1)")
    if val_dataset is not None:
        val_pos = sum(int(g.y.sum().item()) for g in val_dataset)
        val_neg = sum(int((g.y == 0).sum().item()) for g in val_dataset)
        print(f"[*] Val positives:   {val_pos} | Val negatives:   {val_neg:,} (ratio {val_neg / max(val_pos, 1):.0f}:1)")

    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    for i, g in enumerate(dataset[1:], start=2):
        if g.x.shape[1] != num_node_features:
            raise ValueError(f"Graph {i} has {g.x.shape[1]} node features, expected {num_node_features}.")
        if g.edge_attr.shape[1] != num_edge_features:
            raise ValueError(f"Graph {i} has {g.edge_attr.shape[1]} edge features, expected {num_edge_features}.")

    continuous_cols = _get_continuous_edge_cols(num_edge_features)
    edge_norm_stats = compute_global_edge_norm(train_dataset, continuous_cols)
    apply_edge_norm(train_dataset, edge_norm_stats)
    if val_dataset:
        apply_edge_norm(val_dataset, edge_norm_stats)

    prior_pos_rate = train_pos / max(train_pos + train_neg, 1)
    model_config = _build_model_config(args, num_node_features, num_edge_features, prior_pos_rate)

    optimizer_state_dict = None
    if args.resume_from:
        if not Path(args.resume_from).exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")
        model, checkpoint, loaded_config = build_model_from_checkpoint(args.resume_from)
        optimizer_state_dict = checkpoint.get("optimizer_state_dict")
        model_config = loaded_config
        print(f"[*] Resuming from checkpoint: {args.resume_from}")
    else:
        model = FusionPredictor(**model_config)
        print(f"[*] Training from scratch (classifier bias = log-odds prior {math.log(prior_pos_rate / (1 - prior_pos_rate)):.3f}, prior_pos_rate={prior_pos_rate:.5f}).")

    print(f"[*] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[*] Node features: {num_node_features} | Edge features: {num_edge_features}")
    print(f"[*] DataLoader workers: {args.dataloader_workers}")
    print("=" * 50)

    summary = model.train_model_multi_graph(
        train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        batch_size=args.batch_size,
        optimizer_state_dict=optimizer_state_dict,
        dataloader_workers=args.dataloader_workers,
        neg_ratio=args.neg_ratio if args.neg_ratio > 0 else None,
        use_focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        feature_noise_std=args.feature_noise_std,
        select_metric=args.select_metric,
    )

    platt_params = model.fit_platt_scaling(val_dataset) if val_dataset else None

    checkpoint = create_checkpoint(
        model,
        config=model_config,
        optimizer_state_dict=summary["optimizer_state_dict"],
        extra_metadata={
            "epochs_ran": summary["epochs_ran"],
            "best_loss": summary["best_loss"],
            "best_val_f1": summary["best_val_f1"],
            "best_train_f1": summary["best_train_f1"],
            "best_val_thr": summary.get("best_val_thr", 0.5),
            "best_val_auprc": summary.get("best_val_auprc", 0.0),
            "num_graphs": len(dataset),
            "num_train_graphs": len(train_dataset),
            "num_val_graphs": len(val_dataset) if val_dataset else 0,
            "manifest": str(manifest_path),
            "edge_norm_stats": edge_norm_stats,
            "platt_params": platt_params,
        },
    )
    torch.save(checkpoint, output_path)
    print(f"\n[*] Checkpoint saved to {output_path}")
    print(f"[*] Recommended inference threshold (best on val): {summary.get('best_val_thr', 0.5):.4f}")


def main():
    args = _parse_args()
    _configure_torch_threads(args)

    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output_model).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    print(f"[*] Manifest: {manifest_path}")
    print(f"[*] Samples found: {len(rows)}")

    mitelman_db = load_mitelman(args.mitelman_file) if args.mitelman_file else None
    cache_dir = _resolve_cache_dir(args)

    dataset = _build_dataset(rows, args, mitelman_db, cache_dir)
    if not dataset:
        print("[!] No valid graphs were built. Training aborted.")
        sys.exit(1)
    print(f"\n[*] Graphs ready for training: {len(dataset)} / {len(rows)}")

    if args.cv_folds > 1 and len(dataset) > 1:
        _run_cv_folds(dataset, args, manifest_path, output_path)
    else:
        _run_single_split(dataset, args, manifest_path, output_path)


if __name__ == "__main__":
    main()
