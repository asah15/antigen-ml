"""
data.py — FASTA / CSV I/O, cleaning, de-duplication, and a redundancy-aware
train/test split.

Why the redundancy-aware split matters: antigenicity benchmarks are full of
near-identical homologues. A random split leaks homologues across train/test
and inflates every metric. We approximate CD-HIT style clustering with a cheap
greedy k-mer (MinHash-like) clustering so that similar sequences land in the
same fold. Swap in real CD-HIT / MMseqs2 clustering for publication work.

Note: this module hardcodes no biological sequences; any demo input is random.
"""

from __future__ import annotations
import random, hashlib
from dataclasses import dataclass
import numpy as np

VALID = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class Dataset:
    seqs: list[str]
    labels: np.ndarray        # 1 = antigen, 0 = non-antigen
    ids: list[str]

    def __len__(self):
        return len(self.seqs)

    def summary(self):
        n = len(self)
        if n == 0:
            return "empty dataset"
        pos = int(self.labels.sum())
        lens = [len(s) for s in self.seqs]
        return (f"{n} sequences | antigen={pos} ({pos/n:.1%}) "
                f"non-antigen={n-pos} | len min/med/max="
                f"{min(lens)}/{int(np.median(lens))}/{max(lens)}")


# --------------------------------------------------------------------------- IO
def read_fasta(path) -> list[tuple[str, str]]:
    records, name, buf = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(buf)))
                name, buf = line[1:].split()[0] if len(line) > 1 else "seq", []
            elif line:
                buf.append(line)
    if name is not None:
        records.append((name, "".join(buf)))
    return records


def write_fasta(records, path):
    with open(path, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")


def clean_seq(seq: str) -> str:
    return "".join(c for c in seq.upper() if c in VALID)


# ------------------------------------------------------------------ assembling
def build_dataset(pos_records, neg_records, min_len=30, max_len=6000,
                  dedup=True) -> Dataset:
    seqs, labels, ids = [], [], []
    seen = set()
    for recs, lab in ((pos_records, 1), (neg_records, 0)):
        for name, seq in recs:
            s = clean_seq(seq)
            if not (min_len <= len(s) <= max_len):
                continue
            if dedup:
                h = hashlib.md5(s.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
            seqs.append(s); labels.append(lab); ids.append(name)
    return Dataset(seqs, np.asarray(labels, dtype=np.int64), ids)


def load_labeled_csv(path, seq_col="sequence", label_col="label") -> Dataset:
    import pandas as pd
    df = pd.read_csv(path)
    pos = [(f"row{i}", s) for i, s in zip(df.index[df[label_col] == 1],
                                          df.loc[df[label_col] == 1, seq_col])]
    neg = [(f"row{i}", s) for i, s in zip(df.index[df[label_col] == 0],
                                          df.loc[df[label_col] == 0, seq_col])]
    return build_dataset(pos, neg)


# ------------------------------------------------- redundancy-aware clustering
def _kmer_signature(seq: str, k=4, n_hashes=12) -> tuple:
    """Cheap MinHash signature over k-mers for greedy clustering."""
    if len(seq) < k:
        kmers = {seq}
    else:
        kmers = {seq[i:i + k] for i in range(len(seq) - k + 1)}
    sig = []
    for h in range(n_hashes):
        best = min((int(hashlib.md5(f"{h}:{km}".encode()).hexdigest(), 16)
                    for km in kmers), default=0)
        sig.append(best)
    return tuple(sig)


def cluster_labels(seqs, k=4, n_hashes=12) -> np.ndarray:
    """Group sequences sharing a MinHash band into the same cluster (union-find)."""
    sigs = [_kmer_signature(s, k, n_hashes) for s in seqs]
    band_map: dict[tuple, list[int]] = {}
    parent = list(range(len(seqs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, sig in enumerate(sigs):
        for b in range(0, n_hashes, 2):            # bands of 2 hashes
            key = (b, sig[b], sig[b + 1])
            if key in band_map:
                for j in band_map[key]:
                    union(i, j)
                band_map[key].append(i)
            else:
                band_map[key] = [i]
    roots = [find(i) for i in range(len(seqs))]
    remap = {r: c for c, r in enumerate(sorted(set(roots)))}
    return np.asarray([remap[r] for r in roots])


def grouped_split(ds: Dataset, test_frac=0.2, seed=42):
    """Split by cluster so homologues never straddle train/test."""
    clusters = cluster_labels(ds.seqs)
    rng = random.Random(seed)
    uniq = list(set(clusters.tolist()))
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_frac))
    test_clusters = set(uniq[:n_test])
    test_idx = np.array([i for i, c in enumerate(clusters) if c in test_clusters])
    train_idx = np.array([i for i, c in enumerate(clusters) if c not in test_clusters])
    return train_idx, test_idx, clusters


# ------------------------------------------------------- random demo generator
def random_demo_dataset(n_per_class=300, seed=0) -> Dataset:
    """
    Purely synthetic data for PIPELINE TESTING ONLY (no biology).
    Two classes drawn from different random amino-acid frequency vectors so the
    models have *some* separable signal to verify the code end-to-end.
    Replace with real labeled antigen/non-antigen sequences before drawing any
    scientific conclusion.
    """
    rng = np.random.default_rng(seed)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    # two distinct random composition profiles
    p_pos = rng.dirichlet(np.ones(20) * 2.0)
    p_neg = rng.dirichlet(np.ones(20) * 2.0)
    pos, neg = [], []
    for i in range(n_per_class):
        Lp = int(rng.integers(80, 400)); Ln = int(rng.integers(80, 400))
        pos.append((f"demo_pos_{i}", "".join(rng.choice(aa, size=Lp, p=p_pos))))
        neg.append((f"demo_neg_{i}", "".join(rng.choice(aa, size=Ln, p=p_neg))))
    return build_dataset(pos, neg, dedup=False)


if __name__ == "__main__":
    ds = random_demo_dataset(50)
    print(ds.summary())
    tr, te, cl = grouped_split(ds)
    print(f"train={len(tr)} test={len(te)} clusters={len(set(cl.tolist()))}")
