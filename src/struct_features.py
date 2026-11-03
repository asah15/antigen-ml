"""
struct_features.py — Structure-informed descriptors derived purely from sequence.

Since full 3-D coordinates are not always available, all features here are
*predicted* from the amino-acid sequence using well-established look-up tables
and sliding-window algorithms.  For proteins where AlphaFold/PDB structures
ARE available, the companion function ``pdb_features()`` extracts real B-factor,
secondary structure, and per-residue SASA (requires biopython + freesasa).

Feature groups and sizes
------------------------
  SS    : Chou-Fasman secondary-structure propensities (mean per-residue)   (3)
  DISOR : Intrinsic disorder propensity (IUpred-like charge/hydro windows)  (5)
  AMPH  : Amphipathicity & hydrophobic moment (Eisenberg scale)             (5)
  WKDE  : Sliding-window KDE profiles → global stats over hydropathy,
           charge, polarity, flexibility windows                            (12)
  SURF  : Predicted surface-exposure class composition (3 classes)          (3)
─────────────────────────────────────────────────────────────────────────────
  Total                                                                     (28)

Usage
-----
    from struct_features import struct_featurize, struct_feature_names
    x = struct_featurize(seq)          # np.ndarray shape (28,)
    names = struct_feature_names()     # list[str], len 28
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Chou-Fasman propensity tables (Chou & Fasman 1978)
# Pa: α-helix, Pb: β-sheet, Pt: β-turn
# ---------------------------------------------------------------------------
CF_Pa = {
    'A':1.42,'R':0.98,'N':0.67,'D':1.01,'C':0.70,'Q':1.11,'E':1.51,'G':0.57,
    'H':1.00,'I':1.08,'L':1.21,'K':1.16,'M':1.45,'F':1.13,'P':0.57,'S':0.77,
    'T':0.83,'W':1.08,'Y':0.69,'V':1.06,
}
CF_Pb = {
    'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'Q':1.10,'E':0.37,'G':0.75,
    'H':0.87,'I':1.60,'L':1.30,'K':0.74,'M':1.05,'F':1.38,'P':0.55,'S':0.75,
    'T':1.19,'W':1.37,'Y':1.47,'V':1.70,
}
CF_Pt = {
    'A':0.66,'R':0.95,'N':1.56,'D':1.46,'C':1.19,'Q':0.98,'E':0.74,'G':1.56,
    'H':0.95,'I':0.47,'L':0.59,'K':1.01,'M':0.60,'F':0.60,'P':1.52,'S':1.43,
    'T':0.96,'W':0.96,'Y':1.14,'V':0.50,
}

# Eisenberg consensus hydrophobicity scale
EISENBERG = {
    'A': 0.25,'R':-1.76,'N':-0.64,'D':-0.72,'C': 0.04,'Q':-0.69,'E':-0.62,
    'G': 0.16,'H':-0.40,'I': 0.73,'L': 0.53,'K':-1.10,'M': 0.26,'F': 0.61,
    'P':-0.07,'S':-0.26,'T':-0.18,'W': 0.37,'Y': 0.02,'V': 0.54,
}

# Normalized flexiblity (Vihinen 1994); high = disordered
FLEX = {
    'A':0.360,'R':0.530,'N':0.460,'D':0.510,'C':0.350,'Q':0.490,'E':0.500,
    'G':0.540,'H':0.320,'I':0.460,'L':0.370,'K':0.470,'M':0.300,'F':0.310,
    'P':0.510,'S':0.510,'T':0.440,'W':0.310,'Y':0.420,'V':0.390,
}

# Binary charge: +1 for K,R,H; -1 for D,E; 0 otherwise
_CHARGE = {'K':1,'R':1,'H':0.5,'D':-1,'E':-1}

# Predicted surface exposure class (Janin 1979 simplified):
# B = buried, I = intermediate, E = exposed
_SURFACE = {
    'A':'I','R':'E','N':'E','D':'E','C':'B','Q':'E','E':'E','G':'I',
    'H':'E','I':'B','L':'B','K':'E','M':'I','F':'B','P':'I','S':'E',
    'T':'E','W':'B','Y':'E','V':'B',
}

AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


def _safe(seq: str) -> str:
    return "".join(c for c in seq.upper() if c in AA_SET)


# ---------------------------------------------------------------------------
# Individual feature groups
# ---------------------------------------------------------------------------

def ss_propensities(seq: str) -> np.ndarray:
    """Mean Chou-Fasman propensities for α-helix, β-sheet, β-turn (3 values)."""
    if not seq:
        return np.zeros(3, dtype=np.float32)
    pa = np.mean([CF_Pa.get(c, 1.0) for c in seq])
    pb = np.mean([CF_Pb.get(c, 1.0) for c in seq])
    pt = np.mean([CF_Pt.get(c, 1.0) for c in seq])
    return np.asarray([pa, pb, pt], dtype=np.float32)


def disorder_features(seq: str, win=9) -> np.ndarray:
    """
    Simplified IUpred-like disorder propensity using charge and hydrophobicity
    windows (Uversky 2002 two-state criterion + local stats).
    Returns 5 values:
        [mean_local_hydro, std_local_hydro, mean_local_|charge|,
         fraction_disordered_by_CH, global_CH_ratio]
    """
    if len(seq) < win:
        h = [EISENBERG.get(c, 0) for c in seq]
        ch = [abs(_CHARGE.get(c, 0)) for c in seq]
        mh = float(np.mean(h)) if h else 0.0
        mc = float(np.mean(ch)) if ch else 0.0
        return np.asarray([mh, 0.0, mc, 0.0, 0.0], dtype=np.float32)

    local_h, local_c = [], []
    for i in range(len(seq) - win + 1):
        w = seq[i:i + win]
        local_h.append(np.mean([EISENBERG.get(c, 0) for c in w]))
        local_c.append(np.mean([abs(_CHARGE.get(c, 0)) for c in w]))

    local_h = np.array(local_h)
    local_c = np.array(local_c)
    # Uversky boundary: disordered if H < 0.422 (mean hydrophobicity)
    frac_disor = float((local_h < 0.422).mean())
    # CH ratio: low hydro + high charge → disordered
    ch_ratio = float(local_c.mean() / (abs(local_h.mean()) + 1e-6))
    return np.asarray([
        float(local_h.mean()), float(local_h.std()),
        float(local_c.mean()), frac_disor, ch_ratio,
    ], dtype=np.float32)


def amphipathicity(seq: str, win=11) -> np.ndarray:
    """
    Eisenberg hydrophobic moment, mean hydrophobicity, and amphipathicity.
    Returns 5 values: [mean_H, max_moment, mean_moment, std_moment, amphip_ratio]
    """
    if not seq:
        return np.zeros(5, dtype=np.float32)
    H = [EISENBERG.get(c, 0) for c in seq]
    mean_H = float(np.mean(H))
    if len(seq) < win:
        return np.asarray([mean_H, 0, 0, 0, 0], dtype=np.float32)

    moments = []
    # alpha-helix: 100° per residue; beta-sheet: 160° per residue
    for i in range(len(seq) - win + 1):
        h_win = [EISENBERG.get(seq[i + j], 0) for j in range(win)]
        angle_alpha = np.deg2rad(100)
        angles = np.arange(win) * angle_alpha
        mu_sin = np.sum([h * np.sin(a) for h, a in zip(h_win, angles)])
        mu_cos = np.sum([h * np.cos(a) for h, a in zip(h_win, angles)])
        moments.append(np.sqrt(mu_sin ** 2 + mu_cos ** 2) / win)

    moments = np.array(moments)
    max_m = float(moments.max())
    mean_m = float(moments.mean())
    std_m = float(moments.std())
    amphip = mean_m / (abs(mean_H) + 1e-6)
    return np.asarray([mean_H, max_m, mean_m, std_m, amphip], dtype=np.float32)


def window_kde_features(seq: str, win=7) -> np.ndarray:
    """
    Sliding-window statistics over 4 physicochemical profiles:
    hydropathy, net charge, flexibility, polarity.
    For each profile: [mean, std, max, min] over all windows → 4×3 = 12 values.
    (Using mean, std, max only to keep the vector compact.)
    """
    if not seq:
        return np.zeros(12, dtype=np.float32)

    profiles = {
        "hydro": lambda c: EISENBERG.get(c, 0),
        "charge": lambda c: _CHARGE.get(c, 0),
        "flex": lambda c: FLEX.get(c, 0.4),
        "polar": lambda c: 1.0 if c in "RNDQEHKSTY" else 0.0,
    }

    feats = []
    for key, fn in profiles.items():
        vals = [fn(c) for c in seq]
        if len(vals) < win:
            wins = [np.mean(vals)]
        else:
            wins = [np.mean(vals[i:i + win]) for i in range(len(vals) - win + 1)]
        wins = np.array(wins, dtype=np.float32)
        feats += [float(wins.mean()), float(wins.std()), float(wins.max())]

    return np.asarray(feats, dtype=np.float32)


def surface_composition(seq: str) -> np.ndarray:
    """
    Fraction of residues predicted to be buried / intermediate / exposed (3 values).
    """
    counts = {"B": 0, "I": 0, "E": 0}
    for c in seq:
        counts[_SURFACE.get(c, "I")] += 1
    n = len(seq) if seq else 1
    return np.asarray([counts["B"] / n, counts["I"] / n, counts["E"] / n],
                      dtype=np.float32)


# ---------------------------------------------------------------------------
# Combined API
# ---------------------------------------------------------------------------

STRUCT_GROUPS = [
    ("ss", ss_propensities, 3),
    ("disor", disorder_features, 5),
    ("amph", amphipathicity, 5),
    ("wkde", window_kde_features, 12),
    ("surf", surface_composition, 3),
]
STRUCT_DIM = sum(d for _, _, d in STRUCT_GROUPS)   # 28


def struct_featurize(seq: str) -> np.ndarray:
    """Return the full 28-dimensional structural feature vector for one protein."""
    seq = _safe(seq)
    return np.concatenate([fn(seq) for _, fn, _ in STRUCT_GROUPS])


def struct_featurize_many(seqs) -> np.ndarray:
    return np.vstack([struct_featurize(s) for s in seqs])


def struct_feature_names() -> list[str]:
    names = []
    names += ["SS_alpha", "SS_beta", "SS_turn"]
    names += ["DISOR_meanH", "DISOR_stdH", "DISOR_meanC", "DISOR_fracDis", "DISOR_CHratio"]
    names += ["AMPH_meanH", "AMPH_maxMu", "AMPH_meanMu", "AMPH_stdMu", "AMPH_ratio"]
    for profile in ("hydro", "charge", "flex", "polar"):
        names += [f"WKDE_{profile}_mean", f"WKDE_{profile}_std", f"WKDE_{profile}_max"]
    names += ["SURF_buried", "SURF_inter", "SURF_exposed"]
    assert len(names) == STRUCT_DIM, f"Name/dim mismatch: {len(names)} vs {STRUCT_DIM}"
    return names


# ---------------------------------------------------------------------------
# Optional: real structural features from a PDB file
# ---------------------------------------------------------------------------

def pdb_features(pdb_path: str) -> np.ndarray:
    """
    Extract mean B-factor, DSSP secondary-structure fractions, and mean SASA
    from a PDB file.  Requires biopython + freesasa.

    Returns a 7-dim vector:
        [mean_bfactor, norm_bfactor, frac_helix, frac_sheet, frac_coil,
         mean_sasa, frac_exposed_sasa]

    Falls back to zeros if either library is unavailable.
    """
    try:
        from Bio.PDB import PDBParser, DSSP
        import freesasa
    except ImportError:
        return np.zeros(7, dtype=np.float32)

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("prot", pdb_path)
        model = structure[0]

        # B-factors
        bfacs = [atom.get_bfactor() for atom in model.get_atoms()]
        mean_b = float(np.mean(bfacs)) if bfacs else 0.0
        norm_b = float(np.std(bfacs)) if bfacs else 0.0

        # DSSP secondary structure
        dssp = DSSP(model, pdb_path)
        ss_map = {"H": "H", "G": "H", "I": "H",
                  "E": "E", "B": "E",
                  "T": "C", "S": "C", "-": "C", " ": "C"}
        ss_seq = [ss_map.get(dssp[k][2], "C") for k in dssp.keys()]
        n_ss = max(len(ss_seq), 1)
        fH = ss_seq.count("H") / n_ss
        fE = ss_seq.count("E") / n_ss
        fC = ss_seq.count("C") / n_ss

        # SASA via freesasa
        result = freesasa.calcBioPDB(structure)
        total_sasa = result.totalArea()
        rsa_list = []
        for chain in model:
            for res in chain:
                try:
                    rsa_list.append(result.residueAreas()[chain.id][res.id])
                except Exception:
                    pass
        mean_sasa = float(np.mean([r.total for r in rsa_list])) if rsa_list else 0.0
        frac_exp = (float(np.mean([r.total > 25 for r in rsa_list]))
                    if rsa_list else 0.0)

        return np.asarray([mean_b, norm_b, fH, fE, fC, mean_sasa, frac_exp],
                          dtype=np.float32)
    except Exception:
        return np.zeros(7, dtype=np.float32)


if __name__ == "__main__":
    demo = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
    x = struct_featurize(demo)
    print(f"Struct feature vector length: {x.shape[0]}")
    print(f"Names ({len(struct_feature_names())}): {struct_feature_names()[:5]} ...")
    print(f"Values: {x}")
