"""X->bb and X->bs, the parts that do not depend on the sample files (docs/SOPHON_XBB_XBS_PROTOCOL.md).

Sophon (arXiv:2405.12972), Sec. III and App. B.5:
  X->bb, direct (Eq. 4): discr = g(X->bb) / (g(X->bb) + sum_l g(QCD_l)) from the model's own
      softmax outputs; no training. At a vocabulary that merges X->bb with other classes the
      signal node is the merged group (signal_idx takes several nodes).
  X->bs, transfer: the frozen 128-d latent -> MLP (512, 2), ReLU, the first layer preloaded
      from the pretrained model's first output layer (mod.fc.0.0, Linear(128, 512); the paper
      writes Linear(188, 512)); 1 epoch of 2.5 M jets, batch 1024, constant LR 5e-4.
      The optimizer is not stated for this head; Sophon's other trainings use Lookahead
      (k=6, alpha=0.5) over RAdam, so that is the default here [inference].
  Metric (Fig. 2): discovery significance Z (Cowan et al., arXiv:1007.1727: median Z for known b) against the
      SM background efficiency, 40 fb^-1, jets in an m_SD window.

Readers of the samples are not here: their format is unknown until the PI copies them.
"""
from __future__ import annotations

import numpy as np

HEAD_HIDDEN = 512
HEAD_LR = 5e-4
HEAD_BATCH = 1024
HEAD_SAMPLES = 2_500_000
FC0 = "mod.fc.0.0"          # weaver ParT: fc = [Sequential(Linear(128, 512), ReLU, Dropout), Linear(512, K)]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def xbb_discriminant(logits: np.ndarray, signal_idx, qcd_idx) -> np.ndarray:
    """Sophon's Eq. 4 from per-jet logits: the signal node(s) against the sum of the QCD nodes."""
    g = softmax(np.asarray(logits, dtype=np.float64))
    s = g[:, np.atleast_1d(signal_idx)].sum(axis=1)
    q = g[:, np.atleast_1d(qcd_idx)].sum(axis=1)
    if set(np.atleast_1d(signal_idx)) & set(np.atleast_1d(qcd_idx)):
        raise ValueError("a node is both signal and QCD")
    return s / (s + q)


def cowan_z(s, b) -> np.ndarray:
    """Asymptotic discovery significance for s signal over b background events, b > 0."""
    s, b = np.asarray(s, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if np.any(b <= 0):
        raise ValueError("background must be positive")
    return np.sqrt(2 * ((s + b) * np.log1p(s / b) - s))


def in_window(msd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (msd > lo) & (msd < hi)


def z_vs_eps_b(score_s, w_s, score_b, w_b, eps_b) -> dict:
    """Z where the cut on the score keeps a fraction eps_b of the (weighted) background.

    w_s, w_b: each jet's expected yield at the luminosity of interest, after any window cut.
    For each requested eps_b the cut is the score at which the kept background weight first
    reaches eps_b of the total (jets at the cut score are kept); the efficiency it gives is
    returned beside the request, with the signal and background yields it keeps."""
    score_s, w_s = np.asarray(score_s, float), np.asarray(w_s, float)
    score_b, w_b = np.asarray(score_b, float), np.asarray(w_b, float)
    order = np.argsort(-score_b, kind="stable")
    sb, cb = score_b[order], np.cumsum(w_b[order])
    total = cb[-1]
    cuts, got_b = [], []
    for e in np.atleast_1d(eps_b):
        k = min(int(np.searchsorted(cb, e * total, side="left")), len(sb) - 1)
        t = sb[k]
        cuts.append(t)
        got_b.append(w_b[score_b >= t].sum())
    cuts, B = np.array(cuts), np.array(got_b)
    S = np.array([w_s[score_s >= t].sum() for t in cuts])
    return {"eps_b_requested": np.atleast_1d(eps_b).astype(float), "eps_b": B / total, "cut": cuts,
            "S": S, "B": B, "Z": cowan_z(S, B)}


def sophon_head(state_dict: dict | None = None, latent: int = 128):
    """Sophon's transfer head, Linear(latent, 512) -> ReLU -> Linear(512, 2), its first layer
    taken from a pretrained state dict (FC0) when one is given."""
    import torch
    head = torch.nn.Sequential(torch.nn.Linear(latent, HEAD_HIDDEN), torch.nn.ReLU(),
                               torch.nn.Linear(HEAD_HIDDEN, 2))
    if state_dict is not None:
        w, b = state_dict[f"{FC0}.weight"], state_dict[f"{FC0}.bias"]
        if tuple(w.shape) != (HEAD_HIDDEN, latent):
            raise ValueError(f"{FC0}.weight is {tuple(w.shape)}, expected {(HEAD_HIDDEN, latent)}")
        with torch.no_grad():
            head[0].weight.copy_(w)
            head[0].bias.copy_(b)
    return head


class Lookahead:
    """Lookahead (Zhang et al. 2019) over an inner optimizer: every k steps the slow weights
    move alpha of the way to the fast ones and the fast weights restart from them."""

    def __init__(self, inner, k: int = 6, alpha: float = 0.5):
        self.inner, self.k, self.alpha, self.n = inner, k, alpha, 0
        self.slow = [[p.detach().clone() for p in g["params"]] for g in inner.param_groups]

    def zero_grad(self):
        self.inner.zero_grad()

    def step(self):
        import torch
        self.inner.step()
        self.n += 1
        if self.n % self.k:
            return
        with torch.no_grad():
            for g, slow in zip(self.inner.param_groups, self.slow):
                for p, q in zip(g["params"], slow):
                    q.add_(p - q, alpha=self.alpha)
                    p.copy_(q)


def train_head(features: np.ndarray, labels: np.ndarray, state_dict: dict | None, seed: int,
               n_samples: int = HEAD_SAMPLES, batch: int = HEAD_BATCH, lr: float = HEAD_LR):
    """One pass over n_samples jets drawn from (features, labels) with replacement, classes
    1:1 (Sophon's sampling-based reweighting to equal classes), constant LR."""
    import torch
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    labels = np.asarray(labels)
    idx = [np.flatnonzero(labels == c) for c in (0, 1)]
    if any(len(i) == 0 for i in idx):
        raise ValueError("both classes are needed")
    head = sophon_head(state_dict, features.shape[1])
    opt = Lookahead(torch.optim.RAdam(head.parameters(), lr=lr))
    lossf = torch.nn.CrossEntropyLoss()
    X = torch.as_tensor(features, dtype=torch.float32)
    for _ in range(n_samples // batch):
        c = rng.integers(0, 2, batch)
        pick = np.where(c == 0, rng.choice(idx[0], batch), rng.choice(idx[1], batch))
        opt.zero_grad()
        lossf(head(X[pick]), torch.as_tensor(c)).backward()
        opt.step()
    head.eval()
    return head


def head_scores(head, features: np.ndarray) -> np.ndarray:
    import torch
    with torch.no_grad():
        return torch.softmax(head(torch.as_tensor(features, dtype=torch.float32)), 1)[:, 1].numpy()
