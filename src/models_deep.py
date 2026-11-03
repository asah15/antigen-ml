"""
models_deep.py — Deep-learning track for protein antigenicity.

Two options:

1) ProteinCNN  (runs anywhere, no downloads)
   Learned amino-acid embeddings -> multi-kernel 1D convolutions -> global
   pooling -> MLP head. Learns sequence motifs directly from data, giving a
   genuine "representation-learning" comparator to the engineered-feature
   classical track. Trains on CPU for datasets up to a few thousand proteins.

2) esm_embed()  (needs internet: HuggingFace / ESM weights)
   Extracts mean-pooled ESM-2 embeddings so you can train a simple head on a
   pretrained protein language model. This is the state-of-the-art route; it is
   provided ready-to-run for an environment where huggingface.co is reachable.

torch is imported lazily so the rest of the project works even before torch is
installed.
"""

from __future__ import annotations
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
# 0 = pad, 1 = unknown, 2..21 = amino acids
STOI = {a: i + 2 for i, a in enumerate(AA)}
PAD, UNK = 0, 1
VOCAB = len(AA) + 2


def encode(seq: str, max_len: int) -> np.ndarray:
    ids = [STOI.get(c, UNK) for c in seq[:max_len]]
    if len(ids) < max_len:
        ids += [PAD] * (max_len - len(ids))
    return np.asarray(ids, dtype=np.int64)


def encode_batch(seqs, max_len: int) -> np.ndarray:
    return np.vstack([encode(s, max_len) for s in seqs])


# --------------------------------------------------------------------------- CNN
def _build_cnn(max_len, emb_dim=64, channels=128, kernels=(3, 5, 7, 9),
               dropout=0.4, seed=42):
    import torch, torch.nn as nn
    torch.manual_seed(seed)

    class ProteinCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(VOCAB, emb_dim, padding_idx=PAD)
            self.convs = nn.ModuleList([
                nn.Conv1d(emb_dim, channels, k, padding=k // 2) for k in kernels])
            self.bn = nn.ModuleList([nn.BatchNorm1d(channels) for _ in kernels])
            self.drop = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(channels * len(kernels) * 2, 128), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(128, 1))

        def forward(self, x):
            # x: (B, L) -> (B, emb, L)
            h = self.emb(x).transpose(1, 2)
            feats = []
            for conv, bn in zip(self.convs, self.bn):
                c = torch.relu(bn(conv(h)))           # (B, C, L)
                feats.append(torch.max(c, dim=2).values)
                feats.append(torch.mean(c, dim=2))
            z = self.drop(torch.cat(feats, dim=1))
            return self.head(z).squeeze(1)            # (B,)

    return ProteinCNN()


def train_cnn(train_seqs, y_train, val_seqs, y_val, max_len=1000,
              epochs=40, batch_size=32, lr=1e-3, weight_decay=1e-4,
              patience=6, seed=42, verbose=True):
    import torch, torch.nn as nn
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr = torch.tensor(encode_batch(train_seqs, max_len))
    Xva = torch.tensor(encode_batch(val_seqs, max_len))
    ytr = torch.tensor(y_train, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)

    model = _build_cnn(max_len, seed=seed).to(device)
    pos_weight = torch.tensor([(y_train == 0).sum() / max(1, (y_train == 1).sum())],
                              dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = len(Xtr)
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        # validation
        model.eval()
        with torch.no_grad():
            logits = []
            for i in range(0, len(Xva), 256):
                logits.append(model(Xva[i:i + 256].to(device)).cpu())
            p = torch.sigmoid(torch.cat(logits)).numpy()
        auc = roc_auc_score(y_val, p) if len(set(y_val.tolist())) > 1 else 0.5
        if verbose:
            print(f"  epoch {ep:02d} | train_loss {tot/n:.4f} | val_AUC {auc:.4f}")
        if auc > best_auc + 1e-4:
            best_auc, best_state, bad = auc, {k: v.cpu().clone()
                                              for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep} (best val_AUC {best_auc:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def predict_cnn(model, seqs, max_len=1000, batch_size=256):
    import torch
    device = next(model.parameters()).device
    X = torch.tensor(encode_batch(seqs, max_len))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(torch.sigmoid(model(X[i:i + batch_size].to(device))).cpu())
    return torch.cat(out).numpy()


# ------------------------------------------------------- ESM-2 (needs internet)
def esm_embed(seqs, model_name="facebook/esm2_t12_35M_UR50D", batch_size=8,
              max_len=1022):
    """
    Mean-pooled ESM-2 embeddings. Requires `pip install transformers torch` and
    network access to huggingface.co (blocked in some sandboxes). On your own
    machine this gives the pretrained-PLM feature matrix; feed it to any
    sklearn classifier exactly like the engineered features.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    embs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = [s[:max_len] for s in seqs[i:i + batch_size]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len + 2)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc).last_hidden_state           # (B, L, D)
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            embs.append(pooled.cpu().numpy())
    return np.vstack(embs)
