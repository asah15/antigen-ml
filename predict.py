"""
predict.py — Antigenicity inference on new protein sequences.

Usage
-----
    python predict.py --fasta my_proteins.fasta [--model results/best_model.pkl]
                      [--threshold 0.5] [--out predictions.tsv]

If --threshold is not given, the default of 0.5 is used.  Adjust to match the
best_threshold value printed during training if you want maximum MCC.

For models trained with ESM-2 embeddings, the embeddings are computed on-the-fly
for new sequences. This requires internet access and transformers+torch installed.

Output TSV columns
------------------
  protein_id          sequence identifier (from FASTA header, first word)
  length              amino-acid length after cleaning
  antigenicity_score  model probability (0–1); higher = more likely antigen
  prediction          "antigen" / "non-antigen" at the given threshold
  flag                "low_confidence" if 0.4 < score < 0.6, else ""
"""

from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data import read_fasta, clean_seq
from features import featurize_many
from struct_features import struct_featurize_many

SEQ_SETS = ("aac", "dpc", "ctd", "phys")


def extract_features(seqs: list[str], use_sets: list[str],
                     esm_model: str | None = None) -> np.ndarray:
    seq_sets = [s for s in use_sets if s in SEQ_SETS]
    use_struct = "struct" in use_sets
    use_esm = "esm" in use_sets
    parts = []
    if seq_sets:
        parts.append(featurize_many(seqs, sets=tuple(seq_sets)))
    if use_struct:
        parts.append(struct_featurize_many(seqs))
    if use_esm:
        if not esm_model:
            esm_model = "facebook/esm2_t12_35M_UR50D"
        print(f"Computing ESM-2 embeddings ({esm_model}) for {len(seqs)} sequences …",
              file=sys.stderr)
        from models_deep import esm_embed
        parts.append(esm_embed(seqs, model_name=esm_model))
    if not parts:
        raise ValueError("No feature sets found in model metadata")
    return np.hstack(parts)


def load_model(model_path: str) -> dict:
    with open(model_path, "rb") as fh:
        return pickle.load(fh)


def flag(score: float) -> str:
    return "low_confidence" if 0.4 < score < 0.6 else ""


def main():
    ap = argparse.ArgumentParser(description="Predict antigenicity from FASTA")
    ap.add_argument("--fasta", required=True, help="Input FASTA file")
    ap.add_argument("--model", default="results/best_model.pkl",
                    help="Trained model pickle (default: results/best_model.pkl)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold (default: 0.5)")
    ap.add_argument("--out", default="-",
                    help="Output TSV path (default: stdout)")
    ap.add_argument("--min_len", type=int, default=30,
                    help="Skip sequences shorter than this (default: 30)")
    ap.add_argument("--max_len", type=int, default=6000,
                    help="Truncate sequences longer than this (default: 6000)")
    args = ap.parse_args()

    # ---- load model --------------------------------------------------------
    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"Model file not found: {model_path}\n"
                 "Run train.py first, or pass --model <path>.")
    bundle = load_model(str(model_path))
    model      = bundle["model"]
    use_sets   = bundle.get("feature_sets", list(SEQ_SETS) + ["struct"])
    model_name = bundle.get("model_name", "unknown")
    esm_model  = bundle.get("esm_model", None)
    print(f"Model: {model_name}  |  features: {', '.join(use_sets)}",
          file=sys.stderr)

    # ---- read sequences ----------------------------------------------------
    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        sys.exit(f"FASTA file not found: {fasta_path}")
    raw = read_fasta(str(fasta_path))
    if not raw:
        sys.exit("No sequences found in FASTA file")

    ids, seqs = [], []
    skipped = 0
    for name, seq in raw:
        s = clean_seq(seq)[:args.max_len]
        if len(s) < args.min_len:
            skipped += 1
            continue
        ids.append(name)
        seqs.append(s)

    if skipped:
        print(f"Skipped {skipped} sequences shorter than {args.min_len} aa",
              file=sys.stderr)
    print(f"Predicting antigenicity for {len(seqs)} sequence(s) …",
          file=sys.stderr)

    if not seqs:
        sys.exit("No sequences remain after length filtering")

    # ---- extract features & predict ----------------------------------------
    X = extract_features(seqs, use_sets, esm_model=esm_model)
    try:
        probs = model.predict_proba(X)[:, 1]
    except AttributeError:
        scores = model.decision_function(X)
        probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    # ---- write output ------------------------------------------------------
    header = "protein_id\tlength\tantigenicity_score\tprediction\tflag"
    lines  = [header]
    for pid, seq, prob in zip(ids, seqs, probs):
        label = "antigen" if prob >= args.threshold else "non-antigen"
        lines.append(f"{pid}\t{len(seq)}\t{prob:.4f}\t{label}\t{flag(prob)}")

    output = "\n".join(lines) + "\n"
    if args.out == "-":
        sys.stdout.write(output)
    else:
        Path(args.out).write_text(output)
        print(f"Predictions → {args.out}", file=sys.stderr)

    # summary stats
    probs_arr = np.asarray(probs)
    n_pos = int((probs_arr >= args.threshold).sum())
    print(f"Results: {n_pos}/{len(seqs)} predicted antigen "
          f"(threshold={args.threshold:.2f})", file=sys.stderr)


if __name__ == "__main__":
    main()
