"""Phase 4: minimal PARD-SSM. A 4-regime Gaussian HMM (fitted on fold-train rows only, causal forward
filtering) gives regime posteriors + predictive surprise, added as features to the same Random Forest.
Variants: base, ssm, ssm_shuffled (null control: filter run over shuffled row order). No test blocks used."""
import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd, sklearn
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--only", default="")
ap.add_argument("--hmm-iter", type=int, default=15)
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--boot-auc", type=int, default=100)
ap.add_argument("--tag", default="")
args = ap.parse_args()

K = 4
CFG = dict(fpr_budget=0.01, max_benign_train=600_000, block=5000, n_regimes=K, hmm_iter=args.hmm_iter,
           hmm="GaussianHMM diag, 7 core features (signed log1p, train-standardised), random_state=0",
           rf=dict(n_estimators=200, max_samples=0.25, min_samples_leaf=2, n_jobs=-1), seeds=args.seeds, only=args.only)
rid = time.strftime("phase4_%Y%m%d_%H%M%S") + (f"_{args.tag}" if args.tag else "")
R = Path("results") / rid; R.mkdir(parents=True)
(R / "config.json").write_text(json.dumps(CFG, indent=2))
import hmmlearn
(R / "env.json").write_text(json.dumps(dict(python=sys.version, platform=platform.platform(), sklearn=sklearn.__version__,
    numpy=np.__version__, pandas=pd.__version__, hmmlearn=hmmlearn.__version__), indent=2))

# ---- causal forward filter (numba if available, else plain Python)
def _filt(logB, pi, A):
    n, k = logB.shape
    post = np.empty((n, k)); sur = np.empty(n); alpha = pi.copy(); pred = np.empty(k)
    for t in range(n):
        if t == 0:
            for j in range(k): pred[j] = pi[j]
        else:
            for j in range(k):
                s = 0.0
                for i in range(k): s += alpha[i] * A[i, j]
                pred[j] = s
        m = logB[t, 0]
        for j in range(1, k):
            if logB[t, j] > m: m = logB[t, j]
        c = 0.0
        for j in range(k):
            alpha[j] = pred[j] * np.exp(logB[t, j] - m); c += alpha[j]
        if c < 1e-300: c = 1e-300
        for j in range(k):
            alpha[j] /= c; post[t, j] = alpha[j]
        sur[t] = -(np.log(c) + m)
    return post, sur
FILT = _filt
try:
    from numba import njit
    _f = njit(_filt); _f(np.zeros((2, K)), np.full(K, 1.0 / K), np.full((K, K), 1.0 / K)); FILT = _f
    print("numba filter enabled")
except Exception:
    print("numba not available: using plain Python filter (slower)")

# ---- load
D = pd.concat([pd.read_parquet(p) for p in sorted(Path("data/processed").glob("*.parquet"))], ignore_index=True)
S = pd.read_parquet("data/splits/splits_v1.parquet")[["file_order", "row_in_file", "episode", "split", "conflict"]]
D = D.merge(S, on=["file_order", "row_in_file"], how="left", validate="one_to_one")
assert D["split"].notna().all()
META = {"Label", "file_tag", "file_order", "row_in_file", "feat_hash", "dup_in_file",
        "has_invalid", "split", "conflict", "episode"}
FEATS = [c for c in D.columns if c not in META]
fi = {c: i for i, c in enumerate(FEATS)}
X = D[FEATS].to_numpy(dtype="float32")
lab = D["Label"].to_numpy(); y = (lab != "BENIGN").astype(int)
spl, conf, ep = D["split"].to_numpy(), D["conflict"].to_numpy(), D["episode"].to_numpy()
tag, row, hsh = D["file_tag"].to_numpy(), D["row_in_file"].to_numpy(), D["feat_hash"].to_numpy()
del D
N = len(y)
RARE = ["Heartbleed", "Infiltration", "Web Attack - Sql Injection"]
is_rare = np.isin(lab, RARE)
CORE = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Flow Bytes/s",
        "Flow Packets/s", "Flow IAT Mean", "Packet Length Mean"]
assert all(c in fi for c in CORE), [c for c in CORE if c not in fi]
OBS = X[:, [fi[c] for c in CORE]].astype("float64")
OBS = np.sign(OBS) * np.log1p(np.abs(OBS))
FILE_ROWS = {t: np.flatnonzero(tag == t) for t in np.unique(tag)}   # ascending = row order within file

# ---- folds: 14 distinct LFO dev folds + per-episode control (same as Phase 3)
major = sorted(set(lab[y == 1]) - set(RARE))
folds = []
for f in major:
    e = np.unique(ep[lab == f]); assert len(e) == 1
    folds.append((f"family:{f}", "lfo", ep == e[0]))
for t in ["tue", "wed", "thu_am_web"]:
    folds.append((f"day:{t}", "lfo", tag == t))
folds.append(("per_episode:val", "per_episode", np.zeros(N, bool)))
if args.only:
    keys = [k.strip() for k in args.only.split(",")]
    folds = [f for f in folds if any(k in f[0] for k in keys)]
print("folds:", [f[0] for f in folds])

def fit_hmm_and_filter(held, want_shuffled, seed=0):
    rows_h = np.flatnonzero((spl == "train") & ~conf & ~held)
    med = np.nanmedian(OBS[rows_h], axis=0)
    Z = np.where(np.isnan(OBS), med, OBS)
    mu, sd = Z[rows_h].mean(0), Z[rows_h].std(0) + 1e-6
    Z = (Z - mu) / sd
    code = pd.factorize(pd.Series(tag[rows_h]).astype(str) + "|" + pd.Series(ep[rows_h]).astype(str))[0]
    chg = np.flatnonzero(np.diff(code) != 0) + 1
    lengths = np.diff(np.concatenate([[0], chg, [len(code)]]))
    hmm = GaussianHMM(n_components=K, covariance_type="diag", n_iter=args.hmm_iter, tol=1e-3,
                      min_covar=1e-3, random_state=seed)
    hmm.fit(Z[rows_h], lengths)
    var = hmm.covars_
    if var.ndim == 3: var = np.stack([np.diag(v) for v in var])
    logB = np.empty((N, K))
    for k in range(K):
        logB[:, k] = -0.5 * (np.log(2 * np.pi * var[k]).sum() + (((Z - hmm.means_[k]) ** 2) / var[k]).sum(1))
    pi, A = np.asarray(hmm.startprob_, "float64"), np.asarray(hmm.transmat_, "float64")
    out = {}
    for name, shuf in [("ssm", False)] + ([("ssm_shuffled", True)] if want_shuffled else []):
        F = np.empty((N, K + 1), "float32"); rng = np.random.default_rng(123)
        for t, idx in FILE_ROWS.items():
            seq = rng.permutation(idx) if shuf else idx
            post, sur = FILT(np.ascontiguousarray(logB[seq]), pi, A)
            F[seq, :K] = post; F[seq, K] = sur
        out[name] = F
    return out, hmm

def paired_boot(yv, sa, ta, sb, tb, bid, rng):
    o = np.argsort(bid, kind="stable"); yv, sa, sb, bid = yv[o], sa[o], sb[o], bid[o]
    _, st = np.unique(bid, return_index=True); en = np.append(st[1:], len(yv)); nb = len(st)
    def counts(s, t):
        p = s > t; c = lambda m: np.add.reduceat(m.astype(np.int64), st)
        return c(p & (yv == 1)), c(~p & (yv == 1)), c(p & (yv == 0)), c(~p & (yv == 0))
    A, B = counts(sa, ta), counts(sb, tb)
    dr = rng.integers(0, nb, size=(args.boot, nb))
    def rates(c):
        tp, fn, fp, tn = (x[dr].sum(1) for x in c)
        with np.errstate(invalid="ignore", divide="ignore"):
            return tp / (tp + fn), fp / (fp + tn)
    ra, fa = rates(A); rb, fb = rates(B)
    da = []
    for _ in range(args.boot_auc):
        d = rng.integers(0, nb, nb)
        idx = np.concatenate([np.arange(st[i], en[i]) for i in d])
        if len(np.unique(yv[idx])) == 2:
            da.append(roc_auc_score(yv[idx], sb[idx]) - roc_auc_score(yv[idx], sa[idx]))
    q = lambda v: np.nanpercentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)
    (r_lo, r_hi), (f_lo, f_hi), (a_lo, a_hi) = q(rb - ra), q(fb - fa), q(np.array(da))
    return dict(d_rec_lo=r_lo, d_rec_hi=r_hi, d_fpr_lo=f_lo, d_fpr_hi=f_hi, d_auc_lo=a_lo, d_auc_hi=a_hi)

def verdict(r):
    if np.isnan(r["d_rec_lo"]): return "n/a"
    if r["d_rec_lo"] > 0 and r["d_fpr_hi"] <= 0.005: return "better"
    if r["d_rec_hi"] < 0 or r["d_fpr_lo"] > 0.005: return "worse"
    return "no clear change"

variants = ["base", "ssm", "ssm_shuffled"]
rows, delta_rows, prof_rows = [], [], []
rng = np.random.default_rng(0)
for fname, kind, held in folds:
    t0 = time.time()
    tr_all = np.flatnonzero((spl == "train") & ~conf & ~held)
    th = np.unique(hsh[tr_all])
    tr = tr_all[~pd.Series(hsh[tr_all]).duplicated().to_numpy()]
    atk, ben = tr[y[tr] == 1], tr[y[tr] == 0]
    if len(ben) > CFG["max_benign_train"]:
        ben = np.random.RandomState(0).choice(ben, CFG["max_benign_train"], replace=False)
    tr = np.concatenate([atk, ben])
    thr_idx = np.flatnonzero((spl == "val") & ~held & ~conf & (y == 0) & ~np.isin(hsh, th))
    evm = (held & np.isin(spl, ["train", "embargo", "val"])) if kind == "lfo" else (spl == "val")
    ev = np.flatnonzero(evm & ~conf & ~is_rare & ~np.isin(hsh, th))
    yv = y[ev]
    if yv.sum() == 0 or (yv == 0).sum() == 0:
        print(fname, "skipped"); continue
    bid = pd.factorize(pd.Series(tag[ev]).astype(str) + "_" + pd.Series(row[ev] // CFG["block"]).astype(str))[0]
    F, hmm = fit_hmm_and_filter(held, True)
    print(f"\n{fname}: train {len(tr):,}, thr rows {len(thr_idx):,}, eval {len(ev):,} (attacks {int(yv.sum()):,}); "
          f"HMM+filter {time.time() - t0:.0f}s", flush=True)
    for k in range(K):
        prof_rows.append(dict(fold=fname, regime=k, mean_post_benign=float(F["ssm"][ev][yv == 0][:, k].mean()),
                              mean_post_attack=float(F["ssm"][ev][yv == 1][:, k].mean())))
    get = lambda var, r: X[r] if var == "base" else np.hstack([X[r], F[var][r]])
    keep = {}
    for var in variants:
        imp = SimpleImputer(strategy="median").fit(get(var, tr))
        Xtr, ytr = imp.transform(get(var, tr)).astype("float32"), y[tr]
        Xthr, Xev = (imp.transform(get(var, i)).astype("float32") for i in (thr_idx, ev))
        for seed in range(args.seeds):
            clf = RandomForestClassifier(random_state=seed, **CFG["rf"]).fit(Xtr, ytr)
            thr = float(np.quantile(clf.predict_proba(Xthr)[:, 1], 1 - CFG["fpr_budget"]))
            s = clf.predict_proba(Xev)[:, 1]; pred = s > thr
            oth = np.quantile(s[yv == 0], 1 - CFG["fpr_budget"])
            rows.append(dict(fold=fname, variant=var, seed=seed, n_attack=int(yv.sum()), n_benign=int((yv == 0).sum()),
                             roc_auc=roc_auc_score(yv, s), pr_auc=average_precision_score(yv, s),
                             recall_valthr=float(pred[yv == 1].mean()), fpr_valthr=float(pred[yv == 0].mean()),
                             recall_oracle=float((s[yv == 1] > oth).mean()), threshold=thr))
            if seed == 0: keep[var] = (s, thr)
        print(f"  {var} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); d = df[df.fold == fname]
    for var in ["ssm", "ssm_shuffled"]:
        b = d[d.variant == "base"].set_index("seed"); v = d[d.variant == var].set_index("seed")
        r = dict(fold=fname, variant=var, d_recall=(v.recall_valthr - b.recall_valthr).mean(),
                 d_fpr=(v.fpr_valthr - b.fpr_valthr).mean(), d_recall_oracle=(v.recall_oracle - b.recall_oracle).mean(),
                 d_auc=(v.roc_auc - b.roc_auc).mean(), base_recall=b.recall_valthr.mean(), var_recall=v.recall_valthr.mean(),
                 base_fpr=b.fpr_valthr.mean(), var_fpr=v.fpr_valthr.mean())
        r.update(paired_boot(yv, keep["base"][0], keep["base"][1], keep[var][0], keep[var][1], bid, rng))
        r["verdict"] = verdict(r); delta_rows.append(r)
    df.to_csv(R / "phase4_metrics_per_seed.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(R / "phase4_deltas.csv", index=False)
    pd.DataFrame(prof_rows).to_csv(R / "regime_profile.csv", index=False)

res, dl = pd.DataFrame(rows), pd.DataFrame(delta_rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
mean = res.groupby(["fold", "variant"])[["roc_auc", "recall_valthr", "fpr_valthr", "recall_oracle"]].mean().round(3)
print("\n=== means over seeds ===\n", mean.to_string())
cols = ["fold", "variant", "base_recall", "var_recall", "d_recall", "d_rec_lo", "d_rec_hi", "base_fpr", "var_fpr",
        "d_fpr", "d_recall_oracle", "d_auc", "verdict"]
print("\n=== variant minus base (CIs are seed 0, paired block bootstrap) ===\n", dl[cols].round(3).to_string(index=False))
lf = dl[dl.fold != "per_episode:val"]
print("\n=== verdict counts on the LFO folds ===\n", lf.groupby(["variant", "verdict"]).size().unstack(fill_value=0).to_string())
print("\nmean delta recall / delta FPR / delta oracle recall over LFO folds:\n",
      lf.groupby("variant")[["d_recall", "d_fpr", "d_recall_oracle", "d_auc"]].mean().round(4).to_string())
print("\nsaved to", R)