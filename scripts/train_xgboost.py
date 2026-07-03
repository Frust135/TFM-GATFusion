import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_multi_graph import (
    FEATURE_VERSION,
    build_or_load_graph,
    load_manifest,
)
from data_parser.load_mitelman import load_mitelman


def _collect_edge_data(dataset):
    """Stack edge features and labels from all graphs into numpy arrays."""
    all_features = [g.edge_attr.numpy() for g in dataset]
    all_labels = [g.y.numpy() for g in dataset]
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def _graph_worker(args):
    """Build one graph."""
    row, gtf_file, library_type, min_split_reads, mitelman_db, cache_dir = args
    return build_or_load_graph(row, gtf_file, library_type, min_split_reads, mitelman_db, cache_dir)


def _get_feature_names(n_features):
    """Readable names for edge feature columns (see graph_builder.create_graph)."""
    base = [
        "log1p(split_reads)",
        "log1p(discordant_pairs)",
        "split/(discordant+1)",
        "log1p(breakpoint_distance)",
        "log1p(tpm_donor/tpm_acceptor)",
        "is_interchromosomal",
        "junction_type_0",
        "junction_type_1",
        "junction_type_2",
        "junction_type_3",
        "strand_readthrough",
        "strand_tandem_dup",
        "strand_inversion",
        "log1p(repeat_left_len)",
        "log1p(repeat_right_len)",
        "donor_in_exon",
        "acceptor_in_exon",
        "donor_relative_pos",
        "acceptor_relative_pos",
        "log1p(ffpm*1000)",
        "log1p(max_balanced_anchor)",
        "log1p(promiscuity_donor)",
        "log1p(promiscuity_acceptor)",
        "is_read_through",
    ]
    mitelman = ["is_known_pair", "log1p(recurrence)"]
    pair_stats = [
        "log1p(num_junctions)",
        "log1p(max_split_reads_pair)",
        "frac_canonical_junctions",
        "log1p(gene_total_split_donor)",
        "log1p(gene_total_split_acceptor)",
        "frac_canonical_x_both_exonic",
    ]
    names = base + (mitelman if n_features > len(base) + len(pair_stats) else []) + pair_stats
    while len(names) < n_features:
        names.append(f"feature_{len(names)}")
    return names[:n_features]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train XGBoost on edge features (alternative to GNN).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="TSV/CSV manifest (same format as train_multi_graph.py).")
    parser.add_argument("--gtf_file", default=None, help="Default GTF for rows without gtf_file column.")
    parser.add_argument("--output_model", default="checkpoints/xgboost_model.json", help="Path to save the XGBoost model.")
    parser.add_argument("--mitelman_file", default=None, help="Optional Mitelman DB file (must match graph build).")
    parser.add_argument("--graph_cache_dir", default="cache/graphs", help="Graph cache directory. Empty string to disable.")
    parser.add_argument("--no_cache", action="store_true", help="Rebuild all graphs ignoring cache.")
    parser.add_argument("--min_split_reads", type=int, default=2)
    parser.add_argument("--library_type", default="unstranded", choices=["unstranded", "stranded_forward", "stranded_reverse"])
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of graphs for validation.")
    parser.add_argument("--n_estimators", type=int, default=500, help="Number of boosting rounds.")
    parser.add_argument("--max_depth", type=int, default=6, help="Maximum tree depth.")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Boosting learning rate.")
    parser.add_argument("--subsample", type=float, default=1.0, help="Row subsampling ratio per tree.")
    parser.add_argument("--colsample_bytree", type=float, default=1.0, help="Column subsampling ratio per tree.")
    parser.add_argument("--min_child_weight", type=float, default=1.0, help="Minimum sum of instance weight needed in a child node.")
    parser.add_argument("--reg_lambda", type=float, default=1.0, help="L2 regularization on leaf weights.")
    parser.add_argument("--num_workers", type=int, default=1, help="Thread workers for graph building.")
    parser.add_argument("--xgb_threads", type=int, default=0, help="CPU threads for XGBoost (0 = use all available).")
    return parser.parse_args()


def _build_dataset(rows, args, mitelman_db, cache_dir):
    dataset = []
    num_workers = max(1, args.num_workers)
    if num_workers > 1:
        print(f"[*] Building graphs with {num_workers} threads...")
        worker_args = [(row, args.gtf_file, args.library_type, args.min_split_reads, mitelman_db, cache_dir) for row in rows]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_graph_worker, warg) for warg in worker_args]
            for future in as_completed(futures):
                graph = future.result()
                if graph is not None:
                    dataset.append(graph)
    else:
        print("[*] Building graphs sequentially (num_workers=1)...")
        for row in rows:
            graph = build_or_load_graph(row, args.gtf_file, args.library_type, args.min_split_reads, mitelman_db, cache_dir)
            if graph is not None:
                dataset.append(graph)
    return dataset


def _split_train_val(dataset, val_split):
    """Stratified train/val split by graph, so val gets both positive and negative only graphs."""
    rng = random.Random(42)
    pos_graphs = [i for i, g in enumerate(dataset) if g.y.sum().item() > 0]
    neg_graphs = [i for i, g in enumerate(dataset) if g.y.sum().item() == 0]
    rng.shuffle(pos_graphs)
    rng.shuffle(neg_graphs)

    n_val_pos = max(1, int(len(pos_graphs) * val_split)) if pos_graphs else 0
    n_val_neg = max(0, int(len(neg_graphs) * val_split))
    val_idx = set(pos_graphs[:n_val_pos]) | set(neg_graphs[:n_val_neg])

    train_graphs = [dataset[i] for i in range(len(dataset)) if i not in val_idx]
    val_graphs = [dataset[i] for i in sorted(val_idx)]
    return train_graphs, val_graphs


def _evaluate(model, X_train, y_train, X_val, y_val):
    from sklearn.metrics import average_precision_score, classification_report, f1_score, precision_recall_curve

    y_prob_val = model.predict_proba(X_val)[:, 1]
    y_prob_train = model.predict_proba(X_train)[:, 1]
    n_pos_val = int(y_val.sum())

    if n_pos_val > 0:
        precision, recall, thresholds = precision_recall_curve(y_val, y_prob_val)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_thr = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        best_f1_val = float(f1_scores[best_idx])
        auprc = average_precision_score(y_val, y_prob_val)
    else:
        best_thr, best_f1_val, auprc = 0.5, 0.0, 0.0

    y_pred_train = (y_prob_train >= best_thr).astype(int)
    train_f1 = float(f1_score(y_train, y_pred_train, zero_division=0))

    print(f"\n{'=' * 50}")
    print(f"[*] Best validation threshold: {best_thr:.4f}")
    print(f"[*] Val  F1:    {best_f1_val:.4f}")
    print(f"[*] Val  AUPRC: {auprc:.4f}")
    print(f"[*] Train F1:   {train_f1:.4f}")

    if n_pos_val > 0:
        y_pred_val = (y_prob_val >= best_thr).astype(int)
        print(f"\n--- Validation Classification Report (threshold={best_thr:.4f}) ---")
        print(classification_report(y_val, y_pred_val, target_names=["noise", "fusion"], zero_division=0))

    pos_scores_val = y_prob_val[y_val == 1] if n_pos_val > 0 else np.array([])
    neg_scores_val = y_prob_val[y_val == 0] if n_pos_val < len(y_val) else np.array([])
    print("[*] Score distribution (val):")
    if len(pos_scores_val) > 0:
        print(f"    Positives: min={pos_scores_val.min():.4f} mean={pos_scores_val.mean():.4f} max={pos_scores_val.max():.4f}")
    if len(neg_scores_val) > 0:
        print(f"    Negatives: min={neg_scores_val.min():.4f} mean={neg_scores_val.mean():.4f} max={neg_scores_val.max():.4f}")

    importance = model.feature_importances_
    feature_names = _get_feature_names(X_train.shape[1])
    sorted_idx = np.argsort(importance)[::-1]
    print("\n--- Top 10 Feature Importances ---")
    for rank, idx in enumerate(sorted_idx[:10], 1):
        print(f"  {rank:2d}. {feature_names[idx]:35s} {importance[idx]:.4f}")

    return best_thr, best_f1_val, auprc, train_f1


def main():
    args = _parse_args()

    try:
        import xgboost as xgb
    except ImportError:
        print("[!] xgboost not installed. Run: pip install xgboost")
        sys.exit(1)

    manifest_path = Path(args.manifest).resolve()
    rows = load_manifest(manifest_path)
    print(f"[*] Manifest: {manifest_path} ({len(rows)} samples)")

    mitelman_db = load_mitelman(args.mitelman_file) if args.mitelman_file else None

    cache_dir = None
    if args.graph_cache_dir and not args.no_cache:
        cache_dir = Path(args.graph_cache_dir).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = _build_dataset(rows, args, mitelman_db, cache_dir)
    if not dataset:
        print("[!] No valid graphs. Aborting.")
        sys.exit(1)
    print(f"\n[*] Graphs loaded: {len(dataset)}")

    train_graphs, val_graphs = _split_train_val(dataset, args.val_split)
    X_train, y_train = _collect_edge_data(train_graphs)
    X_val, y_val = _collect_edge_data(val_graphs)

    n_pos_train = int(y_train.sum())
    n_neg_train = len(y_train) - n_pos_train
    n_pos_val = int(y_val.sum())
    n_neg_val = len(y_val) - n_pos_val
    scale_pos_weight = n_neg_train / max(n_pos_train, 1)

    print(f"[*] Train: {len(train_graphs)} graphs | {n_pos_train} pos | {n_neg_train:,} neg (ratio {n_neg_train // max(n_pos_train, 1)}:1)")
    print(f"[*] Val:   {len(val_graphs)} graphs | {n_pos_val} pos | {n_neg_val:,} neg")
    print(f"[*] Features per edge: {X_train.shape[1]}")
    print(f"[*] scale_pos_weight: {scale_pos_weight:.1f}")

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=args.xgb_threads if args.xgb_threads > 0 else os.cpu_count(),
        random_state=42,
        early_stopping_rounds=50,
    )

    print(f"\n{'=' * 50}")
    print("[*] Training XGBoost...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    best_thr, best_f1_val, auprc, train_f1 = _evaluate(model, X_train, y_train, X_val, y_val)

    output_path = Path(args.output_model).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))

    metadata = {
        "model_type": "xgboost",
        "feature_version": FEATURE_VERSION,
        "num_edge_features": int(X_train.shape[1]),
        "best_val_thr": best_thr,
        "best_val_f1": best_f1_val,
        "best_val_auprc": auprc,
        "best_train_f1": train_f1,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "reg_lambda": args.reg_lambda,
        "scale_pos_weight": scale_pos_weight,
        "num_graphs": len(dataset),
        "num_train_graphs": len(train_graphs),
        "num_val_graphs": len(val_graphs),
        "has_mitelman": args.mitelman_file is not None,
        "manifest": str(manifest_path),
    }
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[*] Model saved to {output_path}")
    print(f"[*] Metadata saved to {meta_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
