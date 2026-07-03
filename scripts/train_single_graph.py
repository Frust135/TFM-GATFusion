import argparse
import os

import torch

import graph_builder
from data_parser.gtf_dictionary import load_gene_annotations, load_exon_annotations
from data_parser.load_chimeric import load_chimeric
from data_parser.load_discordant_pairs import load_discordant_pairs
from data_parser.load_mitelman import load_mitelman
from data_parser.load_reads_per_gene import load_expression
from model import FusionPredictor, build_model_from_checkpoint, create_checkpoint
from utils.create_train_data import create_train_data


def _parse_positive_fusion(raw):
    """Parse a DONOR:ACCEPTOR string into a (donor, acceptor) tuple."""
    for sep in (":", ",", "\t"):
        if sep in raw:
            donor, acceptor = raw.split(sep, 1)
            donor, acceptor = donor.strip(), acceptor.strip()
            if donor and acceptor:
                return donor, acceptor
    raise ValueError(f"Invalid fusion '{raw}'. Expected DONOR:ACCEPTOR format.")


def _load_positive_fusions(cli_values=None, file_path=None):
    """Load positive fusion pairs from CLI flags and/or a text file."""
    pairs = []
    if cli_values:
        for raw in cli_values:
            pairs.append(_parse_positive_fusion(raw))
    if file_path:
        with open(file_path, "r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    pairs.append(_parse_positive_fusion(stripped))
                except ValueError as exc:
                    raise ValueError(f"{exc} (file {file_path}, line {line_num})") from exc
    if not pairs:
        raise ValueError("No positive fusions provided. Use --positive_fusion or --positive_fusions_file.")
    return list(dict.fromkeys(pairs))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train a gene-fusion GNN on a single RNA-seq sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bam_file", required=True, help="Sorted BAM file")
    parser.add_argument("--chimeric_file", required=True, help="STAR Chimeric.out.junction file")
    parser.add_argument("--reads_per_gene_file", required=True, help="STAR ReadsPerGene.out.tab file")
    parser.add_argument("--gtf_file", required=True, help="Reference annotation GTF")
    parser.add_argument("--output_model", default="fusion_model.pt", help="Path to save the checkpoint")
    parser.add_argument("--resume_from", default=None, help="Checkpoint to continue training from")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--edge_embed_dim", type=int, default=32, help="Dimension of the learned edge embedding produced by the edge encoder MLP.")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--heads", type=int, default=2, help="GATv2 attention heads per layer.")
    parser.add_argument("--num_gnn_layers", type=int, default=2, help="Number of GATv2 layers.")
    parser.add_argument("--no_node_skip", action="store_false", dest="node_skip", help="Disable the raw-node skip pathway into the classifier.")
    parser.set_defaults(node_skip=True)
    parser.add_argument("--positive_fusion", action="append", default=None, help="Known fusion as DONOR:ACCEPTOR. Repeat for multiple fusions.")
    parser.add_argument("--positive_fusions_file", default=None, help="Text file with one DONOR:ACCEPTOR fusion per line.")
    parser.add_argument("--mitelman_file", default=None, help="Optional Mitelman DB flat file (TSV/CSV with gene_a / gene_b columns).")
    return parser.parse_args()


def main():
    args = _parse_args()

    print("\n=== Step 1: Loading STAR data ===")
    discordant_data = load_discordant_pairs(args.bam_file)
    chimeric_data = load_chimeric(args.chimeric_file, min_split_reads=2)
    reads_per_gene_data = load_expression(args.reads_per_gene_file, library_type="unstranded")
    gtf_data = load_gene_annotations(args.gtf_file)
    exon_data = load_exon_annotations(args.gtf_file)
    mitelman_db = load_mitelman(args.mitelman_file) if args.mitelman_file else None

    print("\n=== Step 2: Building graph ===")
    graph = graph_builder.create_graph(
        reads_per_gene_data,
        chimeric_data,
        discordant_data,
        gtf_data,
        mitelman_db=mitelman_db,
        exon_df=exon_data,
    )
    if graph.edge_index.shape[1] == 0:
        raise ValueError("No graph edges were built from the input data.")

    positive_fusions = _load_positive_fusions(cli_values=args.positive_fusion, file_path=args.positive_fusions_file)
    print(f"[*] Labeled fusion pairs: {positive_fusions}")

    graph.y = create_train_data(
        graph.edge_index,
        graph.gene_to_node_id,
        positive_fusions,
        gene_name_to_node_id=graph.gene_name_to_node_id,
    )

    print("\n=== Step 3: Training ===")
    num_node_features = graph.x.shape[1]
    num_edge_features = graph.edge_attr.shape[1]
    num_pos = int(graph.y.sum().item())
    num_neg = int((graph.y == 0).sum().item())
    prior_pos_rate = num_pos / max(num_pos + num_neg, 1) if num_pos > 0 else None

    model_config = {
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
    optimizer_state_dict = None

    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")
        model, checkpoint, loaded_config = build_model_from_checkpoint(args.resume_from, fallback_config=model_config)
        optimizer_state_dict = checkpoint.get("optimizer_state_dict")
        model_config = loaded_config
        if loaded_config["num_node_features"] != num_node_features:
            raise ValueError(f"Feature mismatch: checkpoint has {loaded_config['num_node_features']} node features, current graph has {num_node_features}.")
        if loaded_config["num_edge_features"] != num_edge_features:
            raise ValueError(f"Feature mismatch: checkpoint has {loaded_config['num_edge_features']} edge features, current graph has {num_edge_features}.")
        print(f"[*] Resuming from: {args.resume_from}")
    else:
        model = FusionPredictor(**model_config)
        print("[*] Training from scratch.")

    summary = model.train_model(graph, epochs=args.epochs, lr=args.lr, optimizer_state_dict=optimizer_state_dict)

    print("\n=== Step 4: Saving checkpoint ===")
    checkpoint = create_checkpoint(
        model,
        config=model_config,
        optimizer_state_dict=summary["optimizer_state_dict"],
        extra_metadata={
            "epochs_ran": summary["epochs_ran"],
            "best_loss": summary["best_loss"],
            "positive_fusions": positive_fusions,
        },
    )
    torch.save(checkpoint, args.output_model)
    print(f"[*] Checkpoint saved to {args.output_model}")


if __name__ == "__main__":
    main()
