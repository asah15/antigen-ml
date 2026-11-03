"""
download_data.py — Fetch and prepare labeled antigenicity training data.

TWO MODES — pick the one that fits your situation:

──────────────────────────────────────────────────────────────────────────────
MODE A  (recommended): IEDB bulk export  --iedb_zip / --iedb_csv
──────────────────────────────────────────────────────────────────────────────
1. Open https://www.iedb.org/database_export_v3.php in your browser.
2. Download  "B Cell Epitopes" → bcell_full_v3.zip  (≈ 100-200 MB).
3. Run:
       python src/download_data.py --iedb_zip path/to/bcell_full_v3.zip
   or, if you already unzipped it:
       python src/download_data.py --iedb_csv path/to/bcell_full_v3.csv

The script reads unique source-protein UniProt accessions from the IEDB file,
downloads their sequences from UniProt, and builds the negative set from
non-pathogenic Swiss-Prot proteins.

──────────────────────────────────────────────────────────────────────────────
MODE B : automatic UniProt download (no manual step needed)
──────────────────────────────────────────────────────────────────────────────
       python src/download_data.py

Fetches Swiss-Prot proteins annotated with keyword KW-0045 ("Antigen") from
viral/bacterial organisms as positives, and non-pathogenic reviewed proteins
as negatives.  Requires internet access.

──────────────────────────────────────────────────────────────────────────────
MODE C : offline synthetic data (pipeline testing only)
──────────────────────────────────────────────────────────────────────────────
       python src/download_data.py --demo

Outputs
-------
    data/raw/antigens.fasta
    data/raw/non_antigens.fasta
    data/processed/dataset.csv    ← sequence + label, cleaned, deduplicated
"""

from __future__ import annotations
import argparse, csv, io, random, sys, time, zipfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import urlencode

import numpy as np

ROOT    = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR= ROOT / "data" / "processed"

UNIPROT      = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ACCS = "https://rest.uniprot.org/uniprotkb/accessions"
UNIPROT_F    = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"
VALID     = set("ACDEFGHIKLMNPQRSTVWY")

# UniProt query strings
# taxonomy_id: searches the full lineage (all descendants of the node),
# whereas organism_id: matches only exact nodes — use taxonomy_id for
# superkingdom-level searches like Viruses (10239) and Bacteria (2).
POS_QUERY = (
    "(keyword:KW-0045)"
    " AND (reviewed:true)"
    " AND (taxonomy_id:10239 OR taxonomy_id:2)"
    " AND (length:[50 TO 3000])"
)
NEG_QUERY = (
    "(reviewed:true)"
    " AND (taxonomy_id:9606 OR taxonomy_id:559292 OR taxonomy_id:3702)"
    " AND NOT (keyword:KW-0045)"
    " AND (length:[50 TO 3000])"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def clean(seq: str) -> str:
    return "".join(c for c in seq.upper() if c in VALID)


def parse_fasta(text: str) -> list[tuple[str, str]]:
    records, name, buf = [], None, []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if name:
                records.append((name, "".join(buf)))
            name = line[1:].split()[0]
            buf = []
        elif line:
            buf.append(line)
    if name:
        records.append((name, "".join(buf)))
    return records


def _fetch(url: str, accept: str = "text/plain; format=fasta",
           timeout: int = 60) -> str:
    req = Request(url, headers={
        "Accept": accept,
        "User-Agent": "antigen-ml-research/1.0",
    })
    with urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# IEDB bulk export parser
# ---------------------------------------------------------------------------

# Column names IEDB has used across versions — we try each in order.
# IEDB v3 stores accessions as IRI URLs in "Source Molecule IRI" /
# "Molecule Parent IRI" (e.g. http://www.uniprot.org/uniprot/P04637).
_ACCS_COLS = [
    "Antigen UniProt ID",
    "Antigen -> UniProt ID",
    "Source Molecule Accession Number",
    "Antigen Accession",
    "antigen_uniprot_id",
    # IEDB v3 (bcell_full_v3.csv) — IRI-based columns
    "Source Molecule IRI",
    "Molecule Parent IRI",
]
_ORG_COLS = [
    "Antigen Organism Name",
    "Antigen -> Organism Name",
    "Organism",
    "antigen_organism_name",
]


import re as _re
# UniProt accession regex: 6-char or 10-char formats
# see https://www.uniprot.org/help/accession_numbers
_UNI_ACC_RE = _re.compile(
    r'^[OPQ][0-9][A-Z][A-Z0-9]{2}[0-9]$'          # older 6-char
    r'|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$'  # newer 6- and 10-char
)


def _acc_from_value(val: str) -> str:
    """
    Extract a UniProt accession from:
      - a plain accession  (P04637)
      - a UniProt IRI      (http://www.uniprot.org/uniprot/P04637)
    Non-UniProt IRIs (NCBI, PDB, …) are silently dropped.
    Returns '' for anything that does not match the UniProt accession format.
    """
    val = val.strip().rstrip("/")
    if not val:
        return ""
    if "/" in val:
        # Only accept IRIs from uniprot.org
        if "uniprot" not in val.lower():
            return ""
        val = val.split("/")[-1]
    tok = val.split("-")[0].upper()          # strip isoform suffix (P04637-2)
    return tok if _UNI_ACC_RE.match(tok) else ""


def _find_col(headers: list[str], candidates: list[str]) -> str | None:
    h_lower = {h.lower(): h for h in headers}
    for c in candidates:
        if c in headers:
            return c
        if c.lower() in h_lower:
            return h_lower[c.lower()]
    return None


def parse_iedb_csv(csv_path: Path, max_rows: int = 500_000) -> set[str]:
    """
    Read an IEDB bcell_full_v3.csv and return unique source-protein UniProt
    accessions.

    IEDB v3 exports use a TWO-ROW header:
      row 0 — section/category labels  ("Reference", "Epitope", …)
      row 1 — actual field names        ("Source Molecule IRI", …)

    Critically, column names repeat across sections (e.g. "Source Molecule IRI"
    appears at column indices 20, 36, 62, 84, 124).  csv.DictReader cannot
    handle duplicates correctly, so we work by column INDEX and always take the
    FIRST occurrence — column 20, the epitope's direct source protein.
    """
    accs: set[str] = set()
    with open(csv_path, encoding="utf-8", errors="replace", newline="") as fh:
        lines = [l for l in fh if not l.startswith("#")]

    if len(lines) < 2:
        print("  [warn] IEDB CSV appears empty.")
        return accs

    # Try using row 0, then row 1, as the header row
    for skip in (0, 1):
        header = next(csv.reader([lines[skip]]), [])
        # Find FIRST occurrence of any candidate column name
        acc_idx = None
        for col_name in _ACCS_COLS:
            col_lower = col_name.lower()
            for idx, h in enumerate(header):
                if h == col_name or h.lower() == col_lower:
                    acc_idx = idx
                    print(f"  Using IEDB column [{idx}]: '{h}' "
                          f"(header row {skip})")
                    break
            if acc_idx is not None:
                break

        if acc_idx is not None:
            # Collect ALL column indices that hold UniProt IRIs (the IEDB v3
            # schema repeats "Source Molecule IRI" / "Molecule Parent IRI"
            # across multiple sections — scan every occurrence).
            iri_indices = [
                idx for idx, h in enumerate(header)
                if h in ("Source Molecule IRI", "Molecule Parent IRI")
            ]
            if acc_idx not in iri_indices:
                iri_indices.insert(0, acc_idx)
            print(f"  Scanning {len(iri_indices)} IRI column(s): "
                  f"{iri_indices}")

            data_lines = lines[skip + 1:]
            for i, row in enumerate(csv.reader(data_lines)):
                if i >= max_rows:
                    break
                for col_idx in iri_indices:
                    if col_idx < len(row):
                        tok = _acc_from_value(row[col_idx])
                        if tok:
                            accs.add(tok)
            return accs

    # Still not found — print both header rows to help debug
    print("  [warn] Could not find accession column in IEDB CSV.")
    for skip in (0, 1):
        header = next(csv.reader([lines[skip]]), [])
        print(f"  Row {skip} columns: {header[:15]} …")
    print("  Tip: check that the file is bcell_full_v3.csv (B-cell export)")
    return accs


def load_iedb(path: Path) -> set[str]:
    """Accept .zip or .csv; return set of UniProt accessions."""
    if path.suffix.lower() == ".zip":
        print(f"  Extracting IEDB zip: {path.name} …")
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist()
                         if n.endswith(".csv") and "bcell" in n.lower()]
            if not csv_names:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                sys.exit("No CSV found inside the zip file.")
            csv_name = csv_names[0]
            print(f"  Reading {csv_name} …")
            with zf.open(csv_name) as cf:
                text = cf.read().decode("utf-8", errors="replace")
        # write to temp file for DictReader
        tmp = Path(path.parent / "_iedb_tmp.csv")
        tmp.write_text(text, encoding="utf-8")
        accs = parse_iedb_csv(tmp)
        tmp.unlink(missing_ok=True)
    else:
        accs = parse_iedb_csv(path)
    print(f"  Unique source-protein accessions: {len(accs)}")
    return accs


# ---------------------------------------------------------------------------
# UniProt sequence fetch
# ---------------------------------------------------------------------------

def fetch_uniprot_fasta_paged(query: str, n: int,
                              page_size: int = 200) -> list[tuple[str, str]]:
    """Stream FASTA records from UniProt search (cursor-paged)."""
    all_recs: list[tuple[str, str]] = []
    cursor = None
    while len(all_recs) < n:
        want = min(page_size, n - len(all_recs))
        params: dict = {"query": query, "format": "fasta", "size": want}
        if cursor:
            params["cursor"] = cursor
        url = UNIPROT + "?" + urlencode(params)
        try:
            req = Request(url, headers={
                "Accept": "text/plain; format=fasta",
                "User-Agent": "antigen-ml-research/1.0",
            })
            with urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                link_hdr = resp.headers.get("Link", "")
                cursor = None
                if 'rel="next"' in link_hdr:
                    for part in link_hdr.split(","):
                        if 'rel="next"' in part:
                            cursor = part.split("cursor=")[-1]\
                                        .split("&")[0].split(">")[0]
        except (URLError, Exception) as exc:
            print(f"\n  [warn] UniProt request failed: {exc}")
            break
        recs = parse_fasta(body)
        if not recs:
            break
        all_recs.extend(recs)
        print(f"  fetched {len(all_recs)}/{n} …", end="\r", flush=True)
        if not cursor:
            break
        time.sleep(0.3)
    print()
    return all_recs[:n]


def fetch_uniprot_by_accessions(accs: list[str],
                                batch: int = 200) -> list[tuple[str, str]]:
    """
    Fetch FASTA sequences for a list of UniProt accessions using the
    dedicated /accessions endpoint (handles up to 200 per request).
    """
    all_recs: list[tuple[str, str]] = []
    n = len(accs)
    for i in range(0, n, batch):
        chunk = accs[i:i + batch]
        params = urlencode({
            "accessions": ",".join(chunk),
            "format": "fasta",
            "size": len(chunk),
        })
        url = UNIPROT_ACCS + "?" + params
        try:
            body = _fetch(url)
            all_recs.extend(parse_fasta(body))
        except Exception as exc:
            print(f"  [warn] batch {i//batch+1} failed: {exc}")
        print(f"  fetched {min(i+batch, n)}/{n} accessions …",
              end="\r", flush=True)
        time.sleep(0.3)
    print()
    return all_recs


# ---------------------------------------------------------------------------
# write helpers
# ---------------------------------------------------------------------------

def write_fasta(records: list[tuple[str, str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")
    print(f"  → {path}  ({len(records)} sequences)")


def write_csv(pos_recs, neg_recs, path: Path):
    import pandas as pd
    rows = ([(n, clean(s), 1) for n, s in pos_recs
             if 50 <= len(clean(s)) <= 3000] +
            [(n, clean(s), 0) for n, s in neg_recs
             if 50 <= len(clean(s)) <= 3000])
    df = pd.DataFrame(rows, columns=["id", "sequence", "label"])
    df = df.drop_duplicates(subset="sequence").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    pos = int(df["label"].sum())
    print(f"  CSV: {len(df)} rows | pos={pos} | neg={len(df)-pos}  → {path}")
    return df


# ---------------------------------------------------------------------------
# synthetic fallback
# ---------------------------------------------------------------------------

def write_synthetic(out_csv: Path, n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    aa  = list("ACDEFGHIKLMNPQRSTVWY")
    p_pos = rng.dirichlet(np.ones(20) * 2)
    p_neg = rng.dirichlet(np.ones(20) * 2)
    rows = []
    for i in range(n):
        L = int(rng.integers(80, 400))
        rows.append((f"synthetic_pos_{i}",
                     "".join(rng.choice(aa, L, p=p_pos)), 1))
        rows.append((f"synthetic_neg_{i}",
                     "".join(rng.choice(aa, L, p=p_neg)), 0))
    import pandas as pd
    df = pd.DataFrame(rows, columns=["id", "sequence", "label"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Synthetic dataset written ({len(df)} rows) → {out_csv}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Download antigenicity training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Outputs")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--iedb_zip",
                     help="Path to bcell_full_v3.zip downloaded from IEDB "
                          "(https://www.iedb.org/database_export_v3.php)")
    src.add_argument("--iedb_csv",
                     help="Path to already-extracted bcell_full_v3.csv")
    src.add_argument("--demo", action="store_true",
                     help="Offline synthetic data (pipeline testing only)")
    ap.add_argument("--n_pos",  type=int, default=2000,
                    help="Max antigen sequences (default 2000)")
    ap.add_argument("--n_neg",  type=int, default=2000,
                    help="Max non-antigen sequences (default 2000)")
    ap.add_argument("--balance", action="store_true",
                    help="Down-sample to 1:1 class ratio")
    ap.add_argument("--out",  default=str(PROC_DIR / "dataset.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_csv = Path(args.out)

    # ── Mode C: synthetic ───────────────────────────────────────────────────
    if args.demo:
        print("Demo mode — writing synthetic dataset (no network required)")
        write_synthetic(out_csv, n=400, seed=args.seed)
        return

    # ── Mode A: IEDB bulk file ──────────────────────────────────────────────
    if args.iedb_zip or args.iedb_csv:
        iedb_path = Path(args.iedb_zip or args.iedb_csv)
        if not iedb_path.exists():
            sys.exit(f"File not found: {iedb_path}")
        print(f"\n=== MODE A: IEDB bulk export — {iedb_path.name} ===")
        accs = load_iedb(iedb_path)
        if not accs:
            sys.exit("No accessions found in the IEDB file. "
                     "Check that it is bcell_full_v3.csv/zip.")
        acc_list = random.Random(args.seed).sample(
            sorted(accs), min(args.n_pos, len(accs)))
        print(f"\n=== Fetching {len(acc_list)} antigen sequences from UniProt ===")
        pos_recs_raw = fetch_uniprot_by_accessions(acc_list)
        pos_recs = [(n, clean(s)) for n, s in pos_recs_raw
                    if 50 <= len(clean(s)) <= 3000]
        print(f"  Valid positives: {len(pos_recs)}")
        if not pos_recs:
            sys.exit("UniProt returned no sequences. "
                     "Check your internet connection.")

    # ── Mode B: UniProt KW-0045 ─────────────────────────────────────────────
    else:
        print("\n=== MODE B: UniProt antigen keyword (KW-0045) ===")
        print("  Querying UniProt for viral/bacterial antigens …")
        pos_recs_raw = fetch_uniprot_fasta_paged(POS_QUERY, args.n_pos)
        pos_recs = [(n, clean(s)) for n, s in pos_recs_raw
                    if 50 <= len(clean(s)) <= 3000]
        print(f"  Valid positives: {len(pos_recs)}")
        if not pos_recs:
            print("\n  UniProt returned no results.")
            print("  → Try MODE A: download bcell_full_v3.zip from")
            print("    https://www.iedb.org/database_export_v3.php")
            print("    then run:  python src/download_data.py --iedb_zip <path>")
            sys.exit(1)

    write_fasta(pos_recs, RAW_DIR / "antigens.fasta")

    # ── Negative set (same for both modes) ──────────────────────────────────
    print(f"\n=== Fetching non-antigens from UniProt (human/yeast/plant) ===")
    neg_recs_raw = fetch_uniprot_fasta_paged(NEG_QUERY, args.n_neg)
    pos_ids = {n.split("|")[1] if "|" in n else n for n, _ in pos_recs}
    neg_recs = [(n, clean(s)) for n, s in neg_recs_raw
                if 50 <= len(clean(s)) <= 3000
                and (n.split("|")[1] if "|" in n else n) not in pos_ids]
    print(f"  Valid negatives: {len(neg_recs)}")
    write_fasta(neg_recs, RAW_DIR / "non_antigens.fasta")

    # ── Balance ─────────────────────────────────────────────────────────────
    if args.balance:
        rng = random.Random(args.seed)
        min_n = min(len(pos_recs), len(neg_recs))
        pos_recs = rng.sample(pos_recs, min_n)
        neg_recs = rng.sample(neg_recs, min_n)
        print(f"\nBalanced to {min_n} per class")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    print("\n=== Writing CSV ===")
    write_csv(pos_recs, neg_recs, out_csv)
    print(f"\nDone. Next step:")
    print(f"  python train.py --csv {out_csv} --models rf,xgb")


if __name__ == "__main__":
    main()
