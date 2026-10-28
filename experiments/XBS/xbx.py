"""X->bb and X->bs, the parts that do not depend on the sample files (docs/SOPHON_XBB_XBS_PROTOCOL.md,
docs/PRESPEC_2026-09.md amendment A5).

Sophon (arXiv:2405.12972), Sec. III and App. B.5:
  X->bb, direct (Eq. 4): discr = g(X->bb) / (g(X->bb) + sum_l g(QCD_l)) from the model's own
      softmax outputs; no training. At a vocabulary that merges X->bb with other classes the
      signal node is the merged group (signal_idx takes several nodes).
  X->bs, transfer: the frozen 128-d latent -> MLP (512, 2), ReLU, the first layer preloaded
      from the pretrained model's first output layer (mod.fc.0.0, Linear(128, 512); the paper
      writes Linear(188, 512)); 1 epoch of 2.5 M jets, batch 1024, constant LR 5e-4.
      Where Sophon states nothing, A5 takes its pretraining optimizer (Lookahead k=6,
      alpha=0.5 over RAdam beta=(0.95, 0.999), eps=1e-5), no weight decay, and the dropout
      of 0.1 the pretrained head has after its first layer.
  Metric (Fig. 2): discovery significance Z (Cowan et al., arXiv:1007.1727: median Z for known b) against the
      SM background efficiency, 40 fb^-1, jets in an m_SD window.

A5's per-model number is y = -ln eps_S at a fixed QCD efficiency eps_B* in the window:
1e-3 if the window holds at least 400,000 QCD jets, else 1e-2 (endpoints()).
"""
from __future__ import annotations

import csv
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
HEAD_HIDDEN = 512
HEAD_DROPOUT = 0.1
HEAD_LR = 5e-4
HEAD_BATCH = 1024
HEAD_SAMPLES = 2_500_000
FC0 = "mod.fc.0.0"          # weaver ParT: fc = [Sequential(Linear(128, 512), ReLU, Dropout), Linear(512, K)]
# label maps: (file, the vocabularies it holds a column for)
MAPS = (("rung_label_maps.v1.csv", ("L188", "L162", "R42_Q1", "R16_Q1")),
        ("rand_label_map.v2.csv", tuple(f"RAND2_p{k}" for k in range(1, 6))),
        ("flavour_pair_map.v2.csv", ("FLAV_F0", "FLAV_F1", "FLAV_F1R")))
# Sophon's zero-shot nodes: X->bb by the node holding X->bb (Eq. 4); X->bs, which no class
# names, by the node holding X->bq (Sec. III's control; A5: bq at 188 and 162 classes, the
# merged group at 43 and 17)
ZERO_SHOT = {"xbb": "label_X_bb", "xbq": "label_X_bq"}
A5_QCD_FOR_1E3 = 400_000
EPS_B_GRID = np.logspace(-5, 0, 26)


def node_groups(arm: str) -> dict:
    """{"xbb": node, "xbq": node, "qcd": [QCD-only nodes], "k": classes} of one vocabulary; a
    mass-output arm has its base vocabulary. A node is QCD only if every class in it is QCD
    (experiments/EVAL/anomaly.py node_roles)."""
    col = arm.removesuffix("_MASS")
    f = next((f for f, cols in MAPS if col in cols), None)
    if f is None:
        raise ValueError(f"no label map has a column {col}")
    rows = list(csv.DictReader((REPO / "configs/labelmaps" / f).open()))
    node = {r["class_name"]: int(r[col]) for r in rows}
    members: dict = {}
    for name, n in node.items():
        members.setdefault(n, []).append(name)
    qcd = sorted(n for n, v in members.items() if all(x.startswith("label_QCD_") for x in v))
    return {**{k: node[v] for k, v in ZERO_SHOT.items()}, "qcd": qcd, "k": len(members)}


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


def xbx_columns(logits: np.ndarray, groups: dict) -> dict:
    """The per-jet output-layer columns X->bb and X->bs read, from float32 logits: the log
    probability of the node holding X->bb, of the node holding X->bq, and of the QCD nodes."""
    z = np.asarray(logits, dtype=np.float64)
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    lse = np.log(e.sum(axis=1))
    lp = lambda cols: (np.log(e[:, cols].sum(axis=1)) - lse).astype(np.float32)
    return {"argmax": z.argmax(axis=1).astype(np.int16), "logp_xbb": lp([groups["xbb"]]),
            "logp_xbq": lp([groups["xbq"]]), "logp_qcd": lp(groups["qcd"])}


def zero_shot(logp_sig, logp_qcd) -> np.ndarray:
    """Eq. 4 from the stored log probabilities: P(sig) / (P(sig) + P(QCD))."""
    return 1.0 / (1.0 + np.exp(np.asarray(logp_qcd, np.float64) - np.asarray(logp_sig, np.float64)))


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


def endpoints(score_s, score_b) -> dict:
    """Every number one score gives on the jets inside the window: AUC, A5's working point
    (eps_S at eps_B*, y = -ln eps_S), the efficiency curve Sophon's significance is drawn
    from (eps_S / sqrt(eps_B), the s << b limit of Z up to the yields), and the rejection
    1/eps_B at 50 % and 30 % signal efficiency with the background jets that pass."""
    from sklearn.metrics import roc_auc_score
    s, b = np.asarray(score_s, np.float64), np.asarray(score_b, np.float64)
    star = 1e-3 if b.size >= A5_QCD_FOR_1E3 else 1e-2
    r = z_vs_eps_b(s, np.full(s.size, 1.0 / s.size), b, np.full(b.size, 1.0 / b.size),
                   [star, *EPS_B_GRID])
    eps_s, eps_b = r["S"], r["eps_b"]
    out = {"n_s": int(s.size), "n_b": int(b.size),
           "auc": float(roc_auc_score(np.r_[np.ones(s.size), np.zeros(b.size)], np.r_[s, b])),
           "eps_b_star": star, "eps_b_at_star": float(eps_b[0]), "eps_s_at_star": float(eps_s[0]),
           "neg_ln_eps_s": float(-np.log(eps_s[0])) if eps_s[0] > 0 else None,
           "n_b_pass_at_star": int(round(eps_b[0] * b.size)),
           "curve": {"eps_b": eps_b[1:].tolist(), "eps_s": eps_s[1:].tolist(),
                     "eps_s_over_sqrt_eps_b": (eps_s[1:] / np.sqrt(eps_b[1:])).tolist()}}
    for name, e in (("r50", 0.5), ("r30", 0.3)):
        cut = np.quantile(s, 1.0 - e)
        n_pass = int((b >= cut).sum())
        out[name] = float(b.size / n_pass) if n_pass else None
        out[f"{name}_n_bkg_pass"] = n_pass
    return out


def grid_edges() -> tuple[np.ndarray, np.ndarray]:
    """The (pT, m_SD) reweighting grid of configs/data/JetClassII_base.yaml, read only."""
    import yaml
    w = yaml.safe_load((REPO / "configs/data/JetClassII_base.yaml").read_text())["weights"]["reweight_vars"]
    return np.asarray(w["jet_pt"], np.float64), np.asarray(w["jet_sdmass"], np.float64)


def _cells(pt, msd, edges) -> np.ndarray:
    """Grid cell of each jet, -1 outside the grid."""
    ep, em = edges
    i = np.searchsorted(ep, pt, side="right") - 1
    j = np.searchsorted(em, msd, side="right") - 1
    ok = (i >= 0) & (i < ep.size - 1) & (j >= 0) & (j < em.size - 1)
    return np.where(ok, i * (em.size - 1) + j, -1)


def _take_per_cell(cells, take, rng) -> np.ndarray:
    """Sorted indices of take[c] jets drawn without replacement from each cell c."""
    perm = rng.permutation(cells.size)
    cp = cells[perm]
    order = np.argsort(cp, kind="stable")
    idx, cs = perm[order], cp[order]
    valid = cs >= 0
    idx, cs = idx[valid], cs[valid]
    start = np.searchsorted(cs, np.arange(take.size))
    rank = np.arange(cs.size) - start[cs]
    return np.sort(idx[rank < take[cs]])


def matched_draw(pt_s, msd_s, pt_b, msd_b, n: int, rng, edges=None) -> tuple[np.ndarray, np.ndarray, dict]:
    """A5's draw: up to n signal and n QCD jets, without replacement, with the SAME count in
    every (pT, m_SD) cell of the reweighting grid, as flat as the two pools allow: each cell
    takes min(cap, signal jets, QCD jets) for the smallest cap that reaches n, and the excess
    over n comes off cells at the cap, one jet each, chosen at random. A pool too small for n
    gives every jet the two classes share a cell with."""
    edges = grid_edges() if edges is None else edges
    cs, cb = _cells(pt_s, msd_s, edges), _cells(pt_b, msd_b, edges)
    ncell = (edges[0].size - 1) * (edges[1].size - 1)
    avail = np.minimum(np.bincount(cs[cs >= 0], minlength=ncell), np.bincount(cb[cb >= 0], minlength=ncell))
    cap = None
    if avail.sum() > n:
        lo, hi = 0, int(avail.max())
        while lo < hi:
            mid = (lo + hi) // 2
            lo, hi = (mid + 1, hi) if np.minimum(avail, mid).sum() < n else (lo, mid)
        cap = lo
        take = np.minimum(avail, cap)
        excess = int(take.sum() - n)
        if excess:
            take[rng.choice(np.flatnonzero(take == cap), size=excess, replace=False)] -= 1
    else:
        take = avail
    idx_s, idx_b = _take_per_cell(cs, take, rng), _take_per_cell(cb, take, rng)
    return idx_s, idx_b, {"n_per_class": int(take.sum()), "cap": cap, "cells_used": int((take > 0).sum()),
                          "signal_pool": int((cs >= 0).sum()), "qcd_pool": int((cb >= 0).sum())}


def sophon_head(state_dict: dict | None = None, latent: int = 128):
    """Sophon's transfer head, Linear(latent, 512) -> ReLU -> Dropout(0.1) -> Linear(512, 2), its
    first layer taken from a pretrained state dict (FC0) when one is given."""
    import torch
    head = torch.nn.Sequential(torch.nn.Linear(latent, HEAD_HIDDEN), torch.nn.ReLU(),
                               torch.nn.Dropout(HEAD_DROPOUT), torch.nn.Linear(HEAD_HIDDEN, 2))
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
    """One pass of n_samples jets in the stored order of (features, labels), from the start
    again where the set is smaller (A5: every model receives the same jets in the same order;
    the classes are balanced by matched_draw). The seed sets the second layer and the dropout."""
    import torch
    labels = np.asarray(labels)
    if np.unique(labels).size != 2:
        raise ValueError("both classes are needed")
    torch.manual_seed(seed)
    head = sophon_head(state_dict, features.shape[1])
    head.train()
    opt = Lookahead(torch.optim.RAdam(head.parameters(), lr=lr, betas=(0.95, 0.999), eps=1e-5))
    lossf = torch.nn.CrossEntropyLoss()
    X = torch.as_tensor(features, dtype=torch.float32)
    Y = torch.as_tensor(labels, dtype=torch.long)
    order = torch.as_tensor(np.arange(n_samples) % labels.size)
    for i in range(n_samples // batch):
        pick = order[i * batch:(i + 1) * batch]
        opt.zero_grad()
        lossf(head(X[pick]), Y[pick]).backward()
        opt.step()
    head.eval()
    return head


def head_scores(head, features: np.ndarray) -> np.ndarray:
    import torch
    with torch.no_grad():
        return torch.softmax(head(torch.as_tensor(features, dtype=torch.float32)), 1)[:, 1].numpy()
