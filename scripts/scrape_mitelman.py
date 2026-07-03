import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from itertools import combinations

BASE_URL = "https://mitelmandatabase.isb-cgc.org"
PAGE_QUERY_URL = BASE_URL + "/page_query"
RESULT_URL = BASE_URL + "/result"

CRITERIA = {
    "abnorm_op": "a",
    "break_op": "a",
    "gene_op": "a",
    "op": "M",
    "search_type": "mb",
}

TABLE_ID = "mb_result"
PAGE_SIZE = 5000

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (compatible; MitelmanScraper/1.0; " "+https://github.com/your-org/AI-Fusion)"),
    "Content-Type": "application/x-www-form-urlencoded",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": BASE_URL + "/result",
}


def _post_json(url: str, payload: dict, timeout: int = 60, retries: int = 3):
    encoded = urllib.parse.urlencode(payload).encode()
    req = urllib.request.Request(url, data=encoded, headers=HEADERS)

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
            if attempt < retries:
                wait = 5 * attempt
                print(f"  [warn] Request failed ({exc}); retrying in {wait}s …", file=sys.stderr)
                time.sleep(wait)

    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts") from last_exc


def _init_session(timeout: int = 30):
    payload = {
        "search_type": "mb",
        "genes_mb": "",
        "abNormOptions": "a",
        "brOptions": "a",
        "geneRadios": "a",
    }
    encoded = urllib.parse.urlencode(payload).encode()
    headers = {
        "User-Agent": HEADERS["User-Agent"],
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": BASE_URL + "/mb_search",
    }
    req = urllib.request.Request(RESULT_URL, data=encoded, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            _ = resp.read()  # discard HTML
    except Exception as exc:
        print(f"  [warn] Session init request failed: {exc}", file=sys.stderr)


def fetch_all_records(delay: float = 1.0) -> list[dict]:
    """Retrieve every case from the Mitelman gene-fusion result set"""
    print("Initialising session …")
    _init_session()

    print(f"Fetching page 1 (records 1–{PAGE_SIZE}) …")
    first_page = _post_json(
        PAGE_QUERY_URL,
        {
            "criteria": json.dumps(CRITERIA),
            "table_id": TABLE_ID,
            "start": 0,
            "length": PAGE_SIZE,
            "draw": 1,
        },
    )

    total = int(first_page.get("recordsTotal", 0))
    if total == 0:
        raise RuntimeError("Server returned 0 records. Check the criteria or endpoint.")

    records = list(first_page["data"])
    print(f"  Total records on server: {total:,}")

    draw = 2
    start = PAGE_SIZE
    while start < total:
        end = min(start + PAGE_SIZE, total)
        print(f"Fetching records {start + 1:,}–{end:,} …")
        time.sleep(delay)

        page = _post_json(
            PAGE_QUERY_URL,
            {
                "criteria": json.dumps(CRITERIA),
                "table_id": TABLE_ID,
                "start": start,
                "length": PAGE_SIZE,
                "draw": draw,
            },
        )
        records.extend(page["data"])
        start += PAGE_SIZE
        draw += 1

    print(f"Downloaded {len(records):,} records total.")
    return records


def _parse_gene_short(gene_short: str) -> list[tuple[str, str]]:
    """Parse the GeneShort field of a Mitelman record into a list of directed gene pairs (gene_a, gene_b)"""
    gene_short = gene_short.strip()
    if not gene_short or gene_short in {" ", "N/A", "-"}:
        return []

    pairs: list[tuple[str, str]] = []

    for fusion_str in gene_short.split(","):
        fusion_str = fusion_str.strip()
        if not fusion_str:
            continue

        genes = [g.strip() for g in fusion_str.split("::") if g.strip()]

        if len(genes) < 2:
            continue
        elif len(genes) == 2:
            a, b = sorted(genes)
            pairs.append((a, b))
        else:
            for a, b in combinations(sorted(genes), 2):
                pairs.append((a, b))

    return pairs


def aggregate_pairs(records: list[dict]) -> dict[tuple[str, str], int]:
    """Count how many cases involve each distinct gene pair"""
    counts: dict[tuple[str, str], int] = defaultdict(int)

    for rec in records:
        gene_short = rec.get("GeneShort", "")
        for pair in _parse_gene_short(gene_short):
            norm = tuple(sorted(g.upper() for g in pair))
            counts[norm] += 1

    return counts


def write_tsv(counts: dict[tuple[str, str], int], output_path: str) -> None:
    """Write the aggregated counts to a TSV file"""
    sorted_pairs = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["gene_a", "gene_b", "n_cases"])
        for (gene_a, gene_b), n in sorted_pairs:
            writer.writerow([gene_a, gene_b, n])

    print(f"Wrote {len(sorted_pairs):,} unique gene pairs to: {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=("Scrape the Mitelman Database and produce a TSV file with " "columns gene_a, gene_b, n_cases."))
    p.add_argument(
        "--output",
        default="mitelman_fusions.tsv",
        help="Path for the output TSV file (default: mitelman_fusions.tsv).",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between page requests (default: 1.0).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    records = fetch_all_records(delay=args.delay)
    counts = aggregate_pairs(records)
    write_tsv(counts, args.output)

    top5 = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print("\nTop 5 most recurrent gene pairs:")
    for (a, b), n in top5:
        print(f"  {a}::{b}  →  {n} cases")


if __name__ == "__main__":
    main()
