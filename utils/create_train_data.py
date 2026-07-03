import torch

# Tolerance in base pairs for breakpoint coordinate matching
_BREAKPOINT_TOLERANCE = 100


def create_train_data(edge_index, gene_to_node_id, positive_fusions, gene_name_to_node_id=None, edge_breakpoints=None):
    """Build a tensor aligned with the edge_index.

    Args:
        edge_index:            (2, E) tensor of directed edges.
        gene_to_node_id:       dict mapping Ensembl ID -> node index.
        positive_fusions:      list of tuples:
                               - (donor, acceptor) for gene-level matching
                               - (donor, acceptor, chr_a, pos_a, chr_b, pos_b) for breakpoint-level matching
        gene_name_to_node_id:  optional dict mapping gene symbol -> node index.
                               When provided, gene symbols in positive_fusions are resolved in addition to Ensembl IDs.
        edge_breakpoints:      optional list of dicts (one per edge) containing chr_donor, brkpt_donor, chr_acceptor, brkpt_acceptor.
                               Required for breakpoint-level matching.
    Returns:
        y: float tensor of shape (E,) with 1.0 for fusion edges, 0.0 otherwise.
    """
    lookup = dict(gene_to_node_id)
    if gene_name_to_node_id:
        for name, nid in gene_name_to_node_id.items():
            if name not in lookup:
                lookup[name] = nid

    gene_only_pairs = set()
    breakpoint_fusions = []

    for fusion in positive_fusions:
        gene_a, gene_b = fusion[0], fusion[1]
        if gene_a not in lookup or gene_b not in lookup:
            print(f"  [WARNING] Fusion {gene_a}-{gene_b} was not found in this sample's reads.")
            continue
        id_a, id_b = lookup[gene_a], lookup[gene_b]

        if len(fusion) == 6 and edge_breakpoints is not None:
            _, _, chr_a, pos_a, chr_b, pos_b = fusion
            breakpoint_fusions.append((id_a, id_b, chr_a, pos_a, chr_b, pos_b))
            breakpoint_fusions.append((id_b, id_a, chr_b, pos_b, chr_a, pos_a))
        else:
            gene_only_pairs.add((id_a, id_b))
            gene_only_pairs.add((id_b, id_a))

    y = torch.zeros(edge_index.shape[1], dtype=torch.float)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()

        # For gene matching
        if (src, dst) in gene_only_pairs:
            y[i] = 1.0
            continue

        # For breakpoint matching
        if breakpoint_fusions and edge_breakpoints is not None:
            bp = edge_breakpoints[i]
            for na, nb, ca, pa, cb, pb in breakpoint_fusions:
                if src == na and dst == nb:
                    if bp["chr_donor"] == ca and abs(bp["brkpt_donor"] - pa) <= _BREAKPOINT_TOLERANCE and bp["chr_acceptor"] == cb and abs(bp["brkpt_acceptor"] - pb) <= _BREAKPOINT_TOLERANCE:
                        y[i] = 1.0
                        break

    positives = int(y.sum().item())
    print(f"Labels ready -> {positives} real fusions | {edge_index.shape[1] - positives} noise events.")
    return y
