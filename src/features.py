"""
features.py — Sequence-derived descriptors for protein-level antigenicity.

All features are computed directly from the amino-acid sequence, so the same
code works for any FASTA of viral/bacterial proteins. Groups implemented:

  AAC   : Amino-acid composition                     (20)
  DPC   : Dipeptide composition                       (400)
  CTD   : Composition + Transition over 7 physico-
          chemical property groupings                 (42)
  PHYS  : Global physicochemical descriptors          (~13)

These are the standard descriptor families used by reverse-vaccinology /
antigenicity predictors (VaxiJen, Vaxign-ML, ANTIGENpro use closely related
sets). Everything is vectorised and dependency-light (NumPy only; Biopython
is used opportunistically for a few global descriptors with a pure-Python
fallback).
"""

from __future__ import annotations
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"          # canonical 20
AA_IDX = {a: i for i, a in enumerate(AA)}

# --- physicochemical scales (standard published values) ---------------------
# Kyte-Doolittle hydropathy
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
      'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
      'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
# monoisotopic-ish average residue masses (Da)
MW = {'A':71.08,'R':156.19,'N':114.10,'D':115.09,'C':103.14,'Q':128.13,
      'E':129.12,'G':57.05,'H':137.14,'I':113.16,'L':113.16,'K':128.17,
      'M':131.19,'F':147.18,'P':97.12,'S':87.08,'T':101.10,'W':186.21,
      'Y':163.18,'V':99.13}
# side-chain charge at pH 7
POS = set("KRH"); NEG = set("DE")
AROMATIC = set("FWY"); ALIPHATIC = set("AVLIG")

# CTD: each of 7 properties splits the 20 AAs into 3 groups (Dubchak scheme)
CTD_GROUPS = {
    "hydrophobicity":   ("RKEDQN", "GASTPHY", "CLVIMFW"),
    "normalized_vdwv":  ("GASTPDC", "NVEQIL", "MHKFRYW"),
    "polarity":         ("LIFWCMVY", "PATGS", "HQRKNED"),
    "polarizability":   ("GASDT", "CPNVEQIL", "KMHFRYW"),
    "charge":           ("KR", "ANCQGHILMFPSTWYV", "DE"),
    "secondary_struct": ("EALMQKRH", "VIYCWFT", "GNPSD"),
    "solvent_access":   ("ALFCGIVW", "RKQEND", "MPSTHY"),
}


def _clean(seq: str) -> str:
    """Upper-case, keep only canonical 20 AAs."""
    return "".join(c for c in seq.upper() if c in AA_IDX)


def aac(seq: str) -> np.ndarray:
    v = np.zeros(20, dtype=np.float32)
    for c in seq:
        v[AA_IDX[c]] += 1.0
    n = v.sum()
    return v / n if n else v


def dpc(seq: str) -> np.ndarray:
    v = np.zeros(400, dtype=np.float32)
    for a, b in zip(seq[:-1], seq[1:]):
        v[AA_IDX[a] * 20 + AA_IDX[b]] += 1.0
    n = v.sum()
    return v / n if n else v


def ctd_ct(seq: str) -> np.ndarray:
    """Composition (3) + Transition (3) per property -> 7*6 = 42 features."""
    feats = []
    for groups in CTD_GROUPS.values():
        g2i = {}
        for gi, members in enumerate(groups):
            for m in members:
                g2i[m] = gi
        encoded = [g2i[c] for c in seq if c in g2i]
        L = len(encoded)
        # Composition: fraction of residues in each group
        comp = np.zeros(3, dtype=np.float32)
        for e in encoded:
            comp[e] += 1.0
        comp = comp / L if L else comp
        # Transition: frequency of adjacent group switches (0-1,0-2,1-2)
        trans = np.zeros(3, dtype=np.float32)
        pair_map = {(0, 1): 0, (1, 0): 0, (0, 2): 1, (2, 0): 1, (1, 2): 2, (2, 1): 2}
        for x, y in zip(encoded[:-1], encoded[1:]):
            if x != y:
                trans[pair_map[(x, y)]] += 1.0
        denom = (L - 1) if L > 1 else 1
        trans = trans / denom
        feats.extend(comp.tolist())
        feats.extend(trans.tolist())
    return np.asarray(feats, dtype=np.float32)


def physico(seq: str) -> np.ndarray:
    """Global physicochemical descriptors."""
    L = len(seq)
    if L == 0:
        return np.zeros(13, dtype=np.float32)
    kd = np.mean([KD[c] for c in seq])                       # GRAVY
    mw = sum(MW[c] for c in seq) + 18.02                     # +water
    pos = sum(c in POS for c in seq)
    neg = sum(c in NEG for c in seq)
    net_charge = pos - neg
    arom = sum(c in AROMATIC for c in seq) / L
    alip = sum(c in ALIPHATIC for c in seq) / L
    frac_pos = pos / L
    frac_neg = neg / L
    cys = seq.count("C") / L
    # instability index (Guruprasad) + pI via Biopython if available
    instab, pI = _biopython_global(seq)
    return np.asarray([
        np.log1p(L), mw / 1000.0, kd, net_charge, net_charge / L,
        arom, alip, frac_pos, frac_neg, cys, (pos + neg) / L,
        instab, pI,
    ], dtype=np.float32)


def _biopython_global(seq: str):
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        pa = ProteinAnalysis(seq)
        return float(pa.instability_index()), float(pa.isoelectric_point())
    except Exception:
        # crude fallback so the pipeline never breaks without Biopython
        return 0.0, 7.0


# --- feature-set registry ---------------------------------------------------
FEATURE_SETS = {
    "aac": (aac, 20),
    "dpc": (dpc, 400),
    "ctd": (ctd_ct, 42),
    "phys": (physico, 13),
}


def featurize(seq: str, sets=("aac", "dpc", "ctd", "phys")) -> np.ndarray:
    seq = _clean(seq)
    return np.concatenate([FEATURE_SETS[s][0](seq) for s in sets])


def featurize_many(seqs, sets=("aac", "dpc", "ctd", "phys")) -> np.ndarray:
    return np.vstack([featurize(s, sets) for s in seqs])


def feature_names(sets=("aac", "dpc", "ctd", "phys")):
    names = []
    for s in sets:
        if s == "aac":
            names += [f"AAC_{a}" for a in AA]
        elif s == "dpc":
            names += [f"DPC_{a}{b}" for a in AA for b in AA]
        elif s == "ctd":
            for prop in CTD_GROUPS:
                names += [f"CTD_C_{prop}_{g}" for g in (1, 2, 3)]
                names += [f"CTD_T_{prop}_{t}" for t in ("12", "13", "23")]
        elif s == "phys":
            names += ["PHYS_logLen", "PHYS_MWkDa", "PHYS_GRAVY", "PHYS_netChg",
                      "PHYS_netChgPerRes", "PHYS_aromatic", "PHYS_aliphatic",
                      "PHYS_fracPos", "PHYS_fracNeg", "PHYS_cys", "PHYS_fracChg",
                      "PHYS_instability", "PHYS_pI"]
    return names


if __name__ == "__main__":
    demo = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
    x = featurize(demo)
    print("feature vector length:", x.shape[0])
    print("first 8:", np.round(x[:8], 3))
