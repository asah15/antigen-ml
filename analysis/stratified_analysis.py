"""
stratified_analysis.py — break held-out performance down by source organism,
predicted secondary-structure class, and sequence length.

Reproduces the two supplementary tables in results/supplementary/. Run against a
clone that already holds a trained classical model and its dataset:

    python analysis/stratified_analysis.py \
        --repo . \
        --model results_classical/best_model.pkl \
        --csv data/processed/dataset.csv \
        --outdir results/supplementary

The held-out split is rebuilt deterministically (seed 42) from the same
grouped_split used in training, so the test set matches the benchmark exactly.
Organism is parsed from the UniProt entry-name suffix in the CSV 'id' column
(e.g. ..._PLAF7 -> Plasmodium falciparum); no network access is required.
MCC and sensitivity/specificity are reported at threshold 0.5.
"""
from __future__ import annotations
import argparse, pickle, csv, os, sys
import numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score, matthews_corrcoef, recall_score


def metrics(y, p, thr=0.5):
    n, npos = len(y), int(y.sum())
    both = 0 < npos < n
    pred = (p >= thr).astype(int)
    return {
        "n": n, "pct_antigen": round(100 * npos / n, 1),
        "roc_auc": round(roc_auc_score(y, p), 3) if both else float("nan"),
        "mcc": round(matthews_corrcoef(y, pred), 3) if both else float("nan"),
        "sens": round(recall_score(y, pred, pos_label=1, zero_division=0), 3) if npos else float("nan"),
        "spec": round(recall_score(y, pred, pos_label=0, zero_division=0), 3) if (n - npos) else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--model", default="results_classical/best_model.pkl")
    ap.add_argument("--csv", default="data/processed/dataset.csv")
    ap.add_argument("--outdir", default="results/supplementary")
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(a.repo, "src"))
    from data import load_labeled_csv, grouped_split
    from features import featurize_many
    from struct_features import struct_featurize_many

    df = pd.read_csv(os.path.join(a.repo, a.csv))
    ds = load_labeled_csv(os.path.join(a.repo, a.csv))
    _, te, _ = grouped_split(ds, test_frac=0.2, seed=42)

    seqs = [ds.seqs[i] for i in te]
    y = ds.labels[te].astype(int)
    rows = [int(ds.ids[i][3:]) for i in te]
    orgs = np.array([str(df.loc[i, "id"]).split("_")[-1] if "_" in str(df.loc[i, "id"]) else "NA" for i in rows])
    lens = np.array([len(s) for s in seqs])

    bundle = pickle.load(open(os.path.join(a.repo, a.model), "rb"))
    model, fsets, fn = bundle["model"], bundle["feature_sets"], bundle["feature_names"]
    parts = [featurize_many(seqs, sets=tuple(s for s in ("aac", "dpc", "ctd", "phys") if s in fsets))]
    if "struct" in fsets:
        parts.append(struct_featurize_many(seqs))
    X = np.hstack(parts)
    p = model.predict_proba(X)[:, 1]

    os.makedirs(os.path.join(a.repo, a.outdir), exist_ok=True)

    # S1: structure + length
    diff = X[:, fn.index("SS_alpha")] - X[:, fn.index("SS_beta")]
    ss = np.where(diff > 0.05, "alpha-leaning", np.where(diff < -0.05, "beta-leaning", "balanced"))
    q1, q2 = np.quantile(lens, [1 / 3, 2 / 3])
    ln = np.where(lens <= q1, f"short (<={int(q1)} aa)", np.where(lens <= q2, f"medium ({int(q1)+1}-{int(q2)} aa)", f"long (>{int(q2)} aa)"))
    s1 = os.path.join(a.repo, a.outdir, "stratified_by_structure_and_length.csv")
    with open(s1, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stratum_type", "stratum", "n", "pct_antigen", "test_roc_auc", "mcc_thr0.5", "sensitivity_thr0.5", "specificity_thr0.5"])
        for lab, arr in (("structure", ss), ("length", ln)):
            for c in pd.unique(arr):
                m = metrics(y[arr == c], p[arr == c])
                w.writerow([lab, c, m["n"], m["pct_antigen"], m["roc_auc"], m["mcc"], m["sens"], m["spec"]])

    # S2: per source organism, antigens of that organism vs ALL test negatives
    neg = y == 0
    s2 = os.path.join(a.repo, a.outdir, "stratified_by_organism.csv")
    with open(s2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["organism_mnemonic", "antigens_n", "roc_auc_vs_negatives", "mcc_thr0.5", "sensitivity_thr0.5", "specificity_thr0.5"])
        for org, cnt in Counter(orgs[y == 1]).most_common():
            m = metrics(y[((orgs == org) & (y == 1)) | neg], p[((orgs == org) & (y == 1)) | neg])
            w.writerow([org, cnt, m["roc_auc"], m["mcc"], m["sens"], m["spec"]])

    print("wrote", s1, "and", s2)


if __name__ == "__main__":
    main()
