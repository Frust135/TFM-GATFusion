import csv
from pathlib import Path


class MitelmanDB:
    """Representation of the Mitelman fusion catalogue"""

    def __init__(self, fusion_genes, pair_count, partner_map):
        self.fusion_genes = fusion_genes
        self.pair_count = pair_count
        self.partner_map = partner_map

    def is_fusion_gene(self, gene_symbol):
        """Return True if the gene appears in any Mitelman fusion"""
        return gene_symbol in self.fusion_genes

    def known_pair_count(self, gene_a, gene_b):
        """Return how many times the pair (a, b) has been reported"""
        return self.pair_count.get(frozenset({gene_a, gene_b}), 0)

    def partner_count(self, gene_symbol):
        """Return the number of unique fusion partners a gene has"""
        return len(self.partner_map.get(gene_symbol, set()))

    def __repr__(self):
        return f"MitelmanDB(fusion_genes={len(self.fusion_genes)}, " f"unique_pairs={len(self.pair_count)})"


def load_mitelman(filepath, col_a="gene_a", col_b="gene_b", count_col=None):
    """Parse a Mitelman flat file into a MitelmanDB instance"""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Mitelman file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as fh:
        sample = fh.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = "\t"

    pair_count = {}
    partner_map = {}

    with open(filepath, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(
            (line for line in fh if not line.startswith("#")),
            delimiter=delimiter,
        )
        if reader.fieldnames is None:
            raise ValueError(f"Could not read headers from {filepath}")

        fieldnames_lower = [f.strip().lower() for f in reader.fieldnames]

        def _find_col(name):
            target = name.lower()
            if target in fieldnames_lower:
                return reader.fieldnames[fieldnames_lower.index(target)]
            raise ValueError(f"Column '{name}' not found in {filepath}. " f"Available columns: {list(reader.fieldnames)}")

        real_col_a = _find_col(col_a)
        real_col_b = _find_col(col_b)
        real_count = _find_col(count_col) if count_col else None

        for row in reader:
            gene_a = row[real_col_a].strip()
            gene_b = row[real_col_b].strip()
            if not gene_a or not gene_b:
                continue

            count = 1
            if real_count:
                try:
                    count = max(1, int(float(row[real_count])))
                except (ValueError, TypeError):
                    count = 1

            key = frozenset({gene_a, gene_b})
            pair_count[key] = pair_count.get(key, 0) + count

            partner_map.setdefault(gene_a, set()).add(gene_b)
            partner_map.setdefault(gene_b, set()).add(gene_a)

    fusion_genes = set(partner_map.keys())
    db = MitelmanDB(fusion_genes, pair_count, partner_map)
    print(f"[*] Mitelman DB loaded: {len(fusion_genes)} fusion genes | " f"{len(pair_count)} unique pairs")
    return db
