import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_parser.gtf_dictionary import load_gene_annotations, load_exon_annotations
from data_parser.load_mitelman import load_mitelman
from predict import _annotate_breakpoints_batch, _build_interval_index, _build_or_load_predict_graph


def predict_xgboost(args):
    try:
        import xgboost as xgb
    except ImportError:
        print("[!] xgboost not installed. Run: pip install xgboost")
        sys.exit(1)

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

    print("\n=== Step 3: Loading XGBoost model ===")
    model_path = Path(args.model).resolve()
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    print(f"[*] Model loaded from {model_path}")

    meta_path = model_path.with_suffix(".meta.json")
    metadata = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"[*] Metadata loaded from {meta_path}")

    if args.threshold is None:
        args.threshold = metadata.get("best_val_thr", 0.5)
        print(f"[*] Using threshold from metadata: {args.threshold:.4f}")

    print("\n=== Step 4: Running prediction ===")
    edge_attr = graph.edge_attr.numpy()
    y_prob = model.predict_proba(edge_attr)[:, 1]

    print("[*] Building breakpoint annotation index...")
    gene_index = _build_interval_index(gtf_data)
    exon_index = _build_interval_index(exon_data)

    id_to_gene_id = {v: k for k, v in graph.gene_to_node_id.items()}
    id_to_gene_name = {}
    for _, row in gtf_data.iterrows():
        gene_id = row["gene_id"]
        if gene_id in graph.gene_to_node_id and row["gene_name"]:
            id_to_gene_name[graph.gene_to_node_id[gene_id]] = row["gene_name"]

    breakpoints = [graph.edge_breakpoints[i] for i in range(graph.edge_index.shape[1])]
    donor_chroms = [bp["chr_donor"] for bp in breakpoints]
    donor_pos = [bp["brkpt_donor"] for bp in breakpoints]
    accept_chroms = [bp["chr_acceptor"] for bp in breakpoints]
    accept_pos = [bp["brkpt_acceptor"] for bp in breakpoints]

    print(f"[*] Annotating {len(breakpoints)} breakpoints...")
    donor_annots = _annotate_breakpoints_batch(gene_index, exon_index, donor_chroms, donor_pos)
    accept_annots = _annotate_breakpoints_batch(gene_index, exon_index, accept_chroms, accept_pos)

    results = []
    for i, bp in enumerate(breakpoints):
        src = graph.edge_index[0, i].item()
        dst = graph.edge_index[1, i].item()

        donor_name = id_to_gene_name.get(src, id_to_gene_id.get(src, str(src)))
        acceptor_name = id_to_gene_name.get(dst, id_to_gene_id.get(dst, str(dst)))

        if donor_name == acceptor_name:
            fusion_type = "intra-genic"
        elif bp["chr_donor"] == bp["chr_acceptor"]:
            fusion_type = "intra-chromosomal"
        else:
            fusion_type = "inter-chromosomal"

        results.append(
            {
                "donor_gene": donor_name,
                "acceptor_gene": acceptor_name,
                "score": round(float(y_prob[i]), 4),
                "split_reads": int(bp["split_reads"]),
                "chr_donor": bp["chr_donor"],
                "brkpt_donor": bp["brkpt_donor"],
                "strand_donor": bp["strand_donor"],
                "donor_region": donor_annots[i],
                "chr_acceptor": bp["chr_acceptor"],
                "brkpt_acceptor": bp["brkpt_acceptor"],
                "strand_acceptor": bp["strand_acceptor"],
                "acceptor_region": accept_annots[i],
                "fusion_type": fusion_type,
            }
        )

    df_all = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)

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
        description="Predict gene fusions using a trained XGBoost model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bam_file", required=True, help="Sorted BAM file")
    parser.add_argument("--chimeric_file", required=True, help="STAR Chimeric.out.junction file")
    parser.add_argument("--reads_per_gene_file", required=True, help="STAR ReadsPerGene.out.tab file")
    parser.add_argument("--gtf_file", required=True, help="Reference annotation GTF")
    parser.add_argument("--model", required=True, help="Trained XGBoost model (.json file)")
    parser.add_argument("--threshold", type=float, default=None, help="Minimum score to report a fusion. If omitted, uses " "best_val_thr from the model metadata.")
    parser.add_argument("--top_k", type=int, default=None, help="Report top-K highest-scoring edges.")
    parser.add_argument("--min_split_reads", type=int, default=2)
    parser.add_argument("--library_type", default="unstranded", choices=["unstranded", "stranded_forward", "stranded_reverse"])
    parser.add_argument("--output", default=None, help="TSV file to save predictions")
    parser.add_argument("--graph_cache_dir", default="cache/graphs", help="Directory to cache built graphs. Empty string to disable.")
    parser.add_argument("--no_cache", action="store_true", help="Rebuild graph from scratch, ignoring cache.")
    parser.add_argument("--mitelman_file", default=None, help="Optional Mitelman DB file (must match training).")
    parser.add_argument("--bam_threads", type=int, default=4, help="Decompression threads for BAM reading.")

    args = parser.parse_args()
    predict_xgboost(args)


if __name__ == "__main__":
    main()
