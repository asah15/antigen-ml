"""
train.py — End-to-end antigenicity model training.

Usage
-----
# 1. Use the built-in synthetic demo (no data required)
    python train.py --demo

# 2. Train on a real labeled CSV (from download_data.py)
    python train.py --csv data/processed/dataset.csv

# 3. Train on two separate FASTAs
    python train.py --pos data/raw/antigens.fasta \
                   --neg data/raw/non_antigens.fasta

# 4. Train with ESM-2 protein language model embeddings (needs internet + torch)
    python train.py --csv data/processed/dataset.csv --features esm
    python train.py --csv data/processed/dataset.csv --features aac,dpc,ctd,phys,struct,esm

Options
-------
  --features   Comma-separated subset of: aac,dpc,ctd,phys,struct,esm
               Default: aac,dpc,ctd,phys,struct  (all classical)
               Add 'esm' to include ESM-2 protein language model embeddings.
               Use 'esm' alone to replace classical features entirely.
  --esm_model  ESM-2 HuggingFace model ID (default: facebook/esm2_t12_35M_UR50D)
               Faster / smaller: facebook/esm2_t6_8M_UR50D (320-dim)
               Larger / better:  facebook/esm2_t30_150M_UR50D (640-dim)
               Best quality:     facebook/esm2_t33_650M_UR50D (1280-dim)
  --esm_cache  Path to cache ESM embeddings (.npy). Re-use across runs to skip
               the slow embedding step. Default: results/esm_cache.npy
  --models     Comma-separated subset of: logreg,svm,rf,xgb
               Default: rf,xgb  (fastest + best)
  --out        Output directory (default: results/)
  --cv         Number of cross-validation folds (default: 5)
  --seed       Random seed (default: 42)

Outputs
-------
  results/
    best_model.pkl              sklearn pipeline (best CV score)
    esm_cache.npy               precomputed ESM embeddings (if --features esm)
    summary.csv                 per-model metrics table
    roc_pr.png                  ROC + PR curves
    metric_bars.png             AUC bar chart
    feature_importance.png      (RF/XGB only) top-30 feature importances
"""

from __future__ import annotations
import argparse, pickle, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data import (read_fasta, build_dataset, load_labeled_csv,
                  random_demo_dataset, grouped_split)
from features import featurize_many, feature_names
from struct_features import struct_featurize_many, struct_feature_names
from models_classical import build_model, CLASSICAL_MODELS
from evaluate import metrics, best_threshold, plot_roc_pr, plot_metric_bars


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------

SEQ_SETS = ("aac", "dpc", "ctd", "phys")
VALID_FEAT_SETS = ("aac", "dpc", "ctd", "phys", "struct", "esm")

# ---------------------------------------------------------------------------
# ESM-2 embedding helpers
# ---------------------------------------------------------------------------

def compute_esm_embeddings(seqs: list[str], model_name: str,
                           cache_path: Path | None = None,
                           seq_ids: list[str] | None = None) -> np.ndarray:
    """Compute (or load from cache) mean-pooled ESM-2 embeddings.

    The cache stores a numpy array of shape (N, D) where N == len(seqs).
    A sidecar .txt file stores the sequence IDs used to validate the cache.
    """
    from models_deep import esm_embed

    id_file = None if cache_path is None else cache_path.with_suffix(".ids.txt")
    # Try loading from cache
    if cache_path is not None and cache_path.exists():
        stored_ids = id_file.read_text().splitlines() if id_file and id_file.exists() else []
        if stored_ids == (seq_ids or []):
            print(f"  Loading cached ESM embeddings from {cache_path}")
            return np.load(cache_path)
        else:
            print("  Cache found but sequence IDs differ — recomputing ESM embeddings.")

    print(f"  Computing ESM-2 embeddings ({model_name}) for {len(seqs)} sequences …")
    print("  (This may take several minutes; embeddings will be cached for future runs)")
    t0 = time.time()
    embs = esm_embed(seqs, model_name=model_name)
    elapsed = time.time() - t0
    print(f"  ESM embedding: shape {embs.shape}  ({elapsed:.1f}s)")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embs)
        if seq_ids and id_file:
            id_file.write_text("\n".join(seq_ids))
        print(f"  ESM embeddings cached → {cache_path}")

    return embs


def extract_features(seqs: list[str], use_sets: list[str],
                     esm_embs: np.ndarray | None = None,
                     indices: list[int] | None = None) -> np.ndarray:
    """Build the feature matrix for a subset of sequences.

    Parameters
    ----------
    seqs      : protein sequences for this split
    use_sets  : list of active feature sets
    esm_embs  : precomputed ESM embeddings for the FULL dataset (N, D)
    indices   : integer indices into esm_embs selecting rows for this split
    """
    seq_sets = [s for s in use_sets if s in SEQ_SETS]
    use_struct = "struct" in use_sets
    use_esm = "esm" in use_sets
    parts = []
    if seq_sets:
        parts.append(featurize_many(seqs, sets=tuple(seq_sets)))
    if use_struct:
        parts.append(struct_featurize_many(seqs))
    if use_esm:
        if esm_embs is None or indices is None:
            raise ValueError("ESM embeddings requested but not provided — "
                             "pass esm_embs and indices to extract_features()")
        parts.append(esm_embs[indices])
    if not parts:
        raise ValueError("No feature sets selected. Valid sets: " +
                         ", ".join(VALID_FEAT_SETS))
    return np.hstack(parts)


def all_feature_names(use_sets: list[str], esm_dim: int = 0) -> list[str]:
    seq_sets = [s for s in use_sets if s in SEQ_SETS]
    names = []
    if seq_sets:
        names += feature_names(sets=tuple(seq_sets))
    if "struct" in use_sets:
        names += struct_feature_names()
    if "esm" in use_sets and esm_dim > 0:
        names += [f"esm_{i}" for i in range(esm_dim)]
    return names


# ---------------------------------------------------------------------------
# cross-validation
# ---------------------------------------------------------------------------

def cross_validate(model_name: str, X: np.ndarray, y: np.ndarray,  # noqa: PLR0913
                   clusters: np.ndarray, n_folds: int, seed: int) -> dict:
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_folds)
    fold_aucs, fold_mccs = [], []
    t0 = time.time()
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=clusters), 1):
        clf = build_model(model_name, seed=seed + fold)
        clf.fit(X[tr], y[tr])
        try:
            prob = clf.predict_proba(X[va])[:, 1]
        except AttributeError:
            prob = clf.decision_function(X[va])
            prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-9)
        m = metrics(y[va], prob)
        fold_aucs.append(m["roc_auc"])
        fold_mccs.append(m["mcc"])
    elapsed = time.time() - t0
    return {
        "model": model_name,
        "cv_auc_mean": float(np.mean(fold_aucs)),
        "cv_auc_std":  float(np.std(fold_aucs)),
        "cv_mcc_mean": float(np.mean(fold_mccs)),
        "cv_mcc_std":  float(np.std(fold_mccs)),
        "time_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# feature importance plot
# ---------------------------------------------------------------------------

def plot_importance(model, feat_names: list[str], out_path: Path, top: int = 30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clf = model
    if hasattr(clf, "named_steps"):
        clf = clf.named_steps.get("clf", clf)
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0]) if clf.coef_.ndim > 1 else np.abs(clf.coef_)
    else:
        return  # not supported

    idx = np.argsort(imp)[-top:][::-1]
    top_names = [feat_names[i] for i in idx]
    top_vals  = imp[idx]

    fig, ax = plt.subplots(figsize=(8, 0.35 * top + 1))
    ax.barh(range(top), top_vals[::-1], color="#4C72B0")
    ax.set_yticks(range(top))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Top {top} features")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature importance → {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train antigenicity classifier")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--demo", action="store_true",
                     help="Use synthetic demo data (no files needed)")
    src.add_argument("--csv",  help="Labeled CSV (columns: sequence, label)")
    src.add_argument("--pos",  help="FASTA of antigen proteins")
    ap.add_argument("--neg",   help="FASTA of non-antigen proteins (with --pos)")
    ap.add_argument("--features",
                    default="aac,dpc,ctd,phys,struct",
                    help="Feature groups (comma-separated). Valid: " +
                         ",".join(VALID_FEAT_SETS))
    ap.add_argument("--esm_model",
                    default="facebook/esm2_t12_35M_UR50D",
                    help="ESM-2 HuggingFace model (used when 'esm' in --features)")
    ap.add_argument("--esm_cache",
                    default=None,
                    help="Cache file for ESM embeddings (.npy). Default: "
                         "<out>/esm_cache.npy")
    ap.add_argument("--models",
                    default="rf,xgb",
                    help="Models to train (comma-separated)")
    ap.add_argument("--out",    default="results", help="Output directory")
    ap.add_argument("--cv",     type=int, default=5, help="CV folds")
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    use_sets   = [s.strip() for s in args.features.split(",")]
    model_list = [m.strip() for m in args.models.split(",")]
    bad_feats = [s for s in use_sets if s not in VALID_FEAT_SETS]
    if bad_feats:
        ap.error(f"Unknown feature set(s): {bad_feats}. Valid: {VALID_FEAT_SETS}")
    bad = [m for m in model_list if m not in CLASSICAL_MODELS]
    if bad:
        ap.error(f"Unknown model(s): {bad}. Choose from {CLASSICAL_MODELS}")

    # ---- load dataset ------------------------------------------------------
    print("\n=== Loading dataset ===")
    if args.demo:
        print("  Using synthetic demo dataset (n=300/class)")
        ds = random_demo_dataset(n_per_class=300, seed=args.seed)
    elif args.csv:
        print(f"  CSV: {args.csv}")
        ds = load_labeled_csv(args.csv)
    elif args.pos:
        neg_path = args.neg or ap.error("--neg required with --pos")
        pos = read_fasta(args.pos)
        neg = read_fasta(args.neg)
        ds = build_dataset(pos, neg)
    else:
        ap.error("Provide --demo, --csv, or --pos/--neg")

    print(" ", ds.summary())

    if len(ds) < 20:
        sys.exit("Dataset too small (<20 sequences). Cannot train.")

    # ---- split -------------------------------------------------------------
    print("\n=== Splitting (redundancy-aware) ===")
    train_idx, test_idx, clusters = grouped_split(ds, test_frac=0.2,
                                                  seed=args.seed)
    print(f"  train={len(train_idx)}  test={len(test_idx)}"
          f"  clusters={len(set(clusters.tolist()))}")

    train_seqs = [ds.seqs[i] for i in train_idx]
    test_seqs  = [ds.seqs[i] for i in test_idx]
    y_train = ds.labels[train_idx]
    y_test  = ds.labels[test_idx]
    train_cl = clusters[train_idx]

    # ---- ESM-2 embeddings (precomputed once for the full dataset) ----------
    esm_embs: np.ndarray | None = None
    esm_dim = 0
    if "esm" in use_sets:
        print(f"\n=== ESM-2 embeddings ({args.esm_model}) ===")
        cache_path = Path(args.esm_cache) if args.esm_cache else out_dir / "esm_cache.npy"
        # Use sequence strings as cache IDs (stable across runs with same dataset)
        seq_ids = [f"{i}:{s[:20]}" for i, s in enumerate(ds.seqs)]
        esm_embs = compute_esm_embeddings(ds.seqs, args.esm_model,
                                          cache_path=cache_path, seq_ids=seq_ids)
        esm_dim = esm_embs.shape[1]
        print(f"  ESM embedding dimension: {esm_dim}")

    # ---- features ----------------------------------------------------------
    print(f"\n=== Extracting features ({', '.join(use_sets)}) ===")
    t0 = time.time()
    X_train = extract_features(train_seqs, use_sets,
                               esm_embs=esm_embs, indices=list(train_idx))
    X_test  = extract_features(test_seqs,  use_sets,
                               esm_embs=esm_embs, indices=list(test_idx))
    feat_names_all = all_feature_names(use_sets, esm_dim=esm_dim)
    print(f"  Feature matrix: train {X_train.shape}  test {X_test.shape}"
          f"  ({time.time()-t0:.1f}s)")

    # ---- CV + final fit + test eval ----------------------------------------
    print(f"\n=== Training models: {model_list} ===")
    summary_rows = []
    test_probs   = {}
    fitted_models = {}

    for mname in model_list:
        print(f"\n--- {mname} ---")
        cv_result = cross_validate(mname, X_train, y_train, train_cl,
                                   args.cv, args.seed)
        print(f"  CV AUC: {cv_result['cv_auc_mean']:.4f} ± {cv_result['cv_auc_std']:.4f}"
              f"  MCC: {cv_result['cv_mcc_mean']:.3f}  ({cv_result['time_s']}s)")

        # final fit on full train set
        model = build_model(mname, seed=args.seed)
        model.fit(X_train, y_train)
        fitted_models[mname] = model

        # test set evaluation
        try:
            prob = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            prob = model.decision_function(X_test)
            prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-9)
        test_probs[mname] = prob

        m = metrics(y_test, prob)
        t, mcc_t = best_threshold(y_test, prob)
        print(f"  Test AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}"
              f"  MCC={m['mcc']:.3f}  F1={m['f1']:.3f}"
              f"  Bal-Acc={m['balanced_acc']:.3f}"
              f"  best-thr={t:.2f}(MCC={mcc_t:.3f})")

        row = {"model": mname, **cv_result, **{f"test_{k}": v for k, v in m.items()},
               "best_threshold": t}
        summary_rows.append(row)

    # ---- save best model ---------------------------------------------------
    best_row = max(summary_rows, key=lambda r: r["cv_auc_mean"])
    best_name = best_row["model"]
    best_model = fitted_models[best_name]
    model_path = out_dir / "best_model.pkl"
    esm_meta: dict = {}
    if "esm" in use_sets:
        esm_meta = {
            "esm_model": args.esm_model,
            "esm_cache": str(out_dir / "esm_cache.npy"),
            "esm_dim": esm_dim,
        }
    with open(model_path, "wb") as fh:
        pickle.dump({
            "model": best_model,
            "model_name": best_name,
            "feature_sets": use_sets,
            "feature_names": feat_names_all,
            "seed": args.seed,
            **esm_meta,
        }, fh)
    print(f"\nBest model: {best_name} (CV AUC={best_row['cv_auc_mean']:.4f})"
          f" → {model_path}")

    # ---- save summary CSV --------------------------------------------------
    df_summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Summary → {summary_path}")

    # ---- plots -------------------------------------------------------------
    print("\n=== Generating plots ===")
    if len(set(y_test.tolist())) > 1:
        plot_roc_pr(test_probs, y_test, out_dir / "roc_pr.png")
        print(f"  ROC+PR → {out_dir}/roc_pr.png")
        plot_metric_bars(summary_rows, out_dir / "metric_bars.png",
                         metric="test_roc_auc")
        print(f"  Metric bars → {out_dir}/metric_bars.png")
        plot_importance(best_model, feat_names_all,
                        out_dir / "feature_importance.png")
    else:
        print("  [warn] Only one class in test set — skipping plots")

    print("\n=== Done ===")
    print(f"Results in: {out_dir}/")


if __name__ == "__main__":
    main()
