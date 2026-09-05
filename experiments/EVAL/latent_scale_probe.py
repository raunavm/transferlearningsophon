"""Does the TRUNK carry absolute momentum, or only its inputs?

WHY THIS EXISTS. D1 asserted the trunk "never sees an absolute momentum." G-0
falsified that for the INPUTS: a GBDT on 51 per-jet summary statistics of the
impact-parameter features recovers ln p_T to R^2 = 0.582 and |eta| to 0.823.
That number is a property of what is fed in, not of what the representation
carries forward, and DECISIONS_PENDING item 2 makes measuring the latter a
BLOCKING precondition on publishing any claim about trunk-carried scale.

This script measures it directly, on the frozen 128-d representation the
classifier head actually sees.

READ THE 0.582 CORRECTLY WHILE USING THIS. It is a LOWER bound on input content,
not an upper bound: the GBDT saw only summary statistics and no relative-p_T
feature, so it could not form the per-track join (absolute p_T from d0err) x
(relative p_T from logptrel) that the mechanism actually predicts. An attention
trunk can. A latent R^2 ABOVE 0.582 is therefore not a contradiction.

THE PERMUTATION NULL IS NOT OPTIONAL. audit/pt_recoverability_probe_spec.md:170-172
requires |eta| to clear R^2_shuf by >= 0.05 to count as affirmative evidence, and
requires signed eta -- the negative control -- to stay WITHIN 0.05 of R^2_shuf.
G-0 reported 0.107 for signed eta and never computed R^2_shuf at all, so on the
spec's own terms "the negative control behaves correctly" was never established.
R^2 on a finite test split is biased upward by fitting capacity alone; the null
is what separates signal from that bias.

Run:
  python3 experiments/EVAL/latent_scale_probe.py \
      --features L162=/data/.../features R16_Q1=/data/.../features \
      --out /data/results/eval/latent_scale_v1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Deterministic and arm-independent, exactly as probe.py: the same jets must land
# in the same split for every arm or the comparison is confounded by the split.
SPLIT_SEED = 20260822
MLP_SEEDS = (0, 1, 2)
N_PERM = 10
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def make_splits(n: int, rng_seed: int = SPLIT_SEED):
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)
    return perm[:a], perm[a:b], perm[b:]


def targets(obs: dict) -> dict:
    """The three targets the spec names, and why each is here.

    ln jet_pt   -- the quantity D1's claim is about.
    |jet_eta|   -- the specific pathway: the Delphes resolution table bins
                   d0err/dzerr in |eta|, so recovering |eta| and NOT signed eta
                   is the signature of that mechanism rather than of a generic
                   leak.
    jet_eta     -- NEGATIVE CONTROL. The resolution table is symmetric in eta, so
                   the sign is not encoded. A probe that recovers it is fitting
                   something else, and every other number here is suspect.
    """
    pt, eta = obs["jet_pt"].astype(np.float64), obs["jet_eta"].astype(np.float64)
    assert (pt > 0).all(), "jet_pt has non-positive entries; ln is undefined"
    t = {"ln_jet_pt": np.log(pt), "abs_jet_eta": np.abs(eta), "jet_eta_signed": eta}
    # m_SD / p_T -- the DIMENSIONLESS quantity the trunk is fed (constituents are
    # scaled to jet p_T = 500), and the protected attribute of a post-hoc
    # concept-erasure projection (future/DIRECTION_2.md, D2-A). Its early-kill
    # test is this R^2: near zero means the flat reweighting already removed
    # linearly-decodable m/p_T and there is nothing to erase. Only present when
    # the extraction carried jet_sdmass, so older feature sets still run.
    if "jet_sdmass" in obs:
        t["m_sd_over_pt"] = obs["jet_sdmass"].astype(np.float64) / pt
    return t


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def fit_ridge(Xtr, ytr, Xva, yva, Xte):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    tr, va, te = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best, best_a, best_va = None, None, -np.inf
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(tr, ytr)
        v = r2(yva, m.predict(va))
        if v > best_va:
            best, best_a, best_va = m, a, v
    return best.predict(te), {"alpha": best_a, "val_r2": best_va}


def fit_mlp(Xtr, ytr, Xva, yva, Xte, seeds=MLP_SEEDS):
    """128 -> 256 -> 1. Nonlinear because a linear probe LOWER-BOUNDS what the
    representation holds; a linear null cannot distinguish "absent" from "present
    but not linearly decodable" (D6). The same logic that makes the MLP mandatory
    beside a classification null makes it mandatory here."""
    import torch
    from sklearn.preprocessing import StandardScaler
    xs = StandardScaler().fit(Xtr)
    ys_mu, ys_sd = float(ytr.mean()), float(ytr.std())
    tr = torch.tensor(xs.transform(Xtr), dtype=torch.float32)
    va = torch.tensor(xs.transform(Xva), dtype=torch.float32)
    te = torch.tensor(xs.transform(Xte), dtype=torch.float32)
    ttr = torch.tensor((ytr - ys_mu) / ys_sd, dtype=torch.float32).unsqueeze(1)
    tva = torch.tensor((yva - ys_mu) / ys_sd, dtype=torch.float32).unsqueeze(1)

    preds, meta = [], []
    for s in seeds:
        torch.manual_seed(s)
        net = torch.nn.Sequential(
            torch.nn.Linear(tr.shape[1], 256), torch.nn.ReLU(),
            torch.nn.Dropout(0.1), torch.nn.Linear(256, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = torch.nn.MSELoss()
        best_state, best_va, bad = None, np.inf, 0
        for epoch in range(60):
            net.train()
            perm = torch.randperm(tr.shape[0])
            for i in range(0, tr.shape[0], 4096):
                idx = perm[i:i + 4096]
                opt.zero_grad()
                lossf(net(tr[idx]), ttr[idx]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(lossf(net(va), tva))
            if v < best_va - 1e-5:
                best_va, bad = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 8:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            preds.append(net(te).squeeze(1).numpy() * ys_sd + ys_mu)
        meta.append({"seed": s, "val_mse": best_va})
    return np.mean(preds, axis=0), {"per_seed": meta, "n_seeds": len(seeds)}


def probe_one(F, y, tr, va, te, do_null: bool):
    """One target, both probes, plus the permutation null the spec requires."""
    out = {}
    pr, m = fit_ridge(F[tr], y[tr], F[va], y[va], F[te])
    out["ridge"] = {"r2": r2(y[te], pr), **m}
    pm, m = fit_mlp(F[tr], y[tr], F[va], y[va], F[te])
    out["mlp"] = {"r2": r2(y[te], pm), **m}

    if do_null:
        # Shuffle the TARGET only. Whatever R^2 survives is what this estimator
        # extracts from a representation that cannot possibly encode the label.
        rng = np.random.default_rng(SPLIT_SEED)
        nulls = []
        for _ in range(N_PERM):
            ysh = y.copy()
            rng.shuffle(ysh)
            p, _ = fit_ridge(F[tr], ysh[tr], F[va], ysh[va], F[te])
            nulls.append(r2(ysh[te], p))
        out["null"] = {"r2_shuf_mean": float(np.mean(nulls)),
                       "r2_shuf_std": float(np.std(nulls)), "n_perm": N_PERM}
        out["r2_minus_null"] = out["ridge"]["r2"] - out["null"]["r2_shuf_mean"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True, help="ARM=/path/to/features")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"n_perm": N_PERM, "split_seed": SPLIT_SEED, "arms": {}}

    for spec in args.features:
        if "=" not in spec:
            raise SystemExit(f"FATAL: --features wants ARM=path, got {spec!r}")
        name, path = spec.split("=", 1)
        d = pathlib.Path(path)
        F = np.load(d / "features.npy")
        obs = dict(np.load(d / "observers.npz"))
        missing = [k for k in ("jet_pt", "jet_eta") if k not in obs]
        if missing:
            raise SystemExit(
                f"FATAL: {d}/observers.npz lacks {missing}. Re-extract with those "
                f"in extract_features.py OBSERVERS; present: {sorted(obs)}")
        n = F.shape[0]
        for k, v in obs.items():
            if v.shape[0] != n:
                raise SystemExit(f"FATAL: observer {k} has {v.shape[0]} rows, features {n}")
        tr, va, te = make_splits(n)
        print(f"\n=== {name} ===  {n:,} jets, {F.shape[1]}-d, "
              f"split {tr.size}/{va.size}/{te.size}")

        arm = {}
        for tname, y in targets(obs).items():
            arm[tname] = probe_one(F, y, tr, va, te, do_null=True)
            r = arm[tname]
            print(f"  {tname:16s} ridge R2 {r['ridge']['r2']:+.4f}   "
                  f"mlp R2 {r['mlp']['r2']:+.4f}   "
                  f"null {r['null']['r2_shuf_mean']:+.4f}   "
                  f"ridge-null {r['r2_minus_null']:+.4f}")
        results["arms"][name] = arm

        # The spec's own two conditions, evaluated rather than asserted.
        ctrl = arm["jet_eta_signed"]
        ctrl_ok = abs(ctrl["ridge"]["r2"] - ctrl["null"]["r2_shuf_mean"]) < 0.05
        eta_ok = arm["abs_jet_eta"]["r2_minus_null"] >= 0.05
        arm["spec_checks"] = {
            "negative_control_within_0.05_of_null": bool(ctrl_ok),
            "abs_eta_clears_null_by_0.05": bool(eta_ok)}
        print(f"  spec: negative control {'OK' if ctrl_ok else 'FAILS'}; "
              f"|eta| pathway {'confirmed' if eta_ok else 'not confirmed'}")

    (out / "latent_scale_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}/latent_scale_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
