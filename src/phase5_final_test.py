"""Phase 5: ONE-SHOT final test evaluation of base RF vs RF+SSM on the held-out test blocks.
Frozen configuration from Phases 2-4; nothing is tuned here. Requires --confirm-final and logs every use."""
import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd, sklearn
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--confirm-final", action="store_true")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--only", default="")
ap.add_argument("--hmm-iter", type=int, default=15)
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--boot-auc", type=int, default=100)
ap.add_argument("--tag", default="")
args = ap.parse_args()
if not args.confirm_final:
    sys.exit("Refusing to touch the test blocks without --confirm-final (one-shot evaluation).")

K, GAP, MIN_ALERTS = 4, 500, 3
CFG = dict(fpr_budget=0.01, max_benign_train=600_000, block=5000, n_regimes=K, hmm_iter=args.hmm_iter, seeds=args.seeds,
           incident=f"attack rows <= {GAP} rows apart form one incident; detected if >= {MIN_ALERTS} flows alerted",
           rf=dict(n_estimators=200, max_samples=0.25, min_samples_leaf=2, n_jobs=-1), only=args.only)
rid = time.strftime("final_%Y%m%d_%H%M%S") + (f"_{args.tag}" if args.tag else "")
R = Path("results") / rid; R.mkdir(parents=True)
(R / "config.json").write_text(json.dumps(CFG, indent=2))
import hmmlearn
(R / "env.json").write_text(json.dumps(dict(python=sys.version, platform=platform.platform(), sklearn=sklearn.__version__,
    numpy=np.__version__, pandas=pd.__version__, hmmlearn=hmmlearn.__version__), indent=2))
with open("results/test_access_log.csv", "a") as f:
    f.write(f"{time.ctime()},{rid}\n")

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
except Exception:
    print("numba not available: plain Python filter")

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
OBS = X[:, [fi[c] for c in CORE]].astype("float64")
OBS = np.sign(OBS) * np.log1p(np.abs(OBS))
FILE_ROWS = {t: np.flatnonzero(tag == t) for t in np.unique(tag)}

major = sorted(set(lab[y == 1]) - set(RARE))
folds = []
for f in major:
    e = np.unique(ep[lab == f]); assert len(e) == 1
    folds.append((f"family:{f}", "lfo", ep == e[0]))
for t in ["tue", "wed", "thu_am_web"]:
    folds.append((f"day:{t}", "lfo", tag == t))
folds.append(("per_episode:test", "per_episode", np.zeros(N, bool)))
if args.only:
    keys = [k.strip() for k in args.only.split(",")]
    folds = [f for f in folds if any(k in f[0] for k in keys)]
print("folds:", [f[0] for f in folds])

def ssm_features(held, seed=0):
    rows_h = np.flatnonzero((spl == "train") & ~conf & ~held)
    med = np.nanmedian(OBS[rows_h], axis=0)
    Z = np.where(np.isnan(OBS), med, OBS)
    mu, sd = Z[rows_h].mean(0), Z[rows_h].std(0) + 1e-6
    Z = (Z - mu) / sd
    code = pd.factorize(pd.Series(tag[rows_h]).astype(str) + "|" + pd.Series(ep[rows_h]).astype(str))[0]
    chg = np.flatnonzero(np.diff(code) != 0) + 1
    lengths = np.diff(np.concatenate([[0], chg, [len(code)]]))
    hmm = GaussianHMM(n_components=K, covariance_type="diag", n_iter=args.hmm_iter, tol=1e-3, min_covar=1e-3, random_state=seed)
    hmm.fit(Z[rows_h], lengths)
    var = hmm.covars_
    if var.ndim == 3: var = np.stack([np.diag(v) for v in var])
    logB = np.empty((N, K))
    for k in range(K):
        logB[:, k] = -0.5 * (np.log(2 * np.pi * var[k]).sum() + (((Z - hmm.means_[k]) ** 2) / var[k]).sum(1))
    pi, A = np.asarray(hmm.startprob_, "float64"), np.asarray(hmm.transmat_, "float64")
    F = np.empty((N, K + 1), "float32")
    for t, idx in FILE_ROWS.items():
        post, sur = FILT(np.ascontiguousarray(logB[idx]), pi, A)
        F[idx, :K] = post; F[idx, K] = sur
    return F

def boot_ci(yv, s, thr, bid, rng, other=None):
    """Block-bootstrap 95% CIs for recall, FPR, AUC (and paired deltas vs `other` = (scores, thr))."""
    o = np.argsort(bid, kind="stable"); yv, s, bid = yv[o], s[o], bid[o]
    if other is not None: so = other[0][o]
    _, st = np.unique(bid, return_index=True); en = np.append(st[1:], len(yv)); nb = len(st)
    cnt = lambda m: np.add.reduceat(m.astype(np.int64), st)
    def parts(sc, t):
        p = sc > t
        return cnt(p & (yv == 1)), cnt(~p & (yv == 1)), cnt(p & (yv == 0)), cnt(~p & (yv == 0))
    dr = rng.integers(0, nb, size=(args.boot, nb))
    def rates(c):
        tp, fn, fp, tn = (x[dr].sum(1) for x in c)
        with np.errstate(invalid="ignore", divide="ignore"):
            return tp / (tp + fn), fp / (fp + tn)
    r, f = rates(parts(s, thr))
    aucs, dau = [], []
    for _ in range(args.boot_auc):
        d = rng.integers(0, nb, nb); idx = np.concatenate([np.arange(st[i], en[i]) for i in d])
        if len(np.unique(yv[idx])) == 2:
            a = roc_auc_score(yv[idx], s[idx]); aucs.append(a)
            if other is not None: dau.append(a - roc_auc_score(yv[idx], so[idx]))
    q = lambda v: tuple(np.nanpercentile(v, [2.5, 97.5])) if len(v) else (np.nan, np.nan)
    out = dict(zip(["recall_lo", "recall_hi", "fpr_lo", "fpr_hi", "auc_lo", "auc_hi"], [*q(r), *q(f), *q(aucs)]))
    if other is not None:
        r2, f2 = rates(parts(so, other[1]))
        out.update(zip(["d_recall_lo", "d_recall_hi", "d_fpr_lo", "d_fpr_hi", "d_auc_lo", "d_auc_hi"],
                       [*q(r - r2), *q(f - f2), *q(np.array(dau))]))
    return out

def incident_stats(ev, yv, pred):
    n_inc = n_det = 0
    for t in np.unique(tag[ev]):
        m = (tag[ev] == t) & (yv == 1)
        if not m.any(): continue
        rr, pp = row[ev][m], pred[m]; o = np.argsort(rr); rr, pp = rr[o], pp[o]
        grp = np.concatenate([[0], np.cumsum(np.diff(rr) > GAP)])
        for g in np.unique(grp):
            n_inc += 1; n_det += int(pp[grp == g].sum() >= MIN_ALERTS)
    return n_inc, n_det

rows, fam_rows, ci_rows = [], [], []
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
    evm = (held & (spl == "test")) if kind == "lfo" else (spl == "test")
    ev = np.flatnonzero(evm & ~conf & ~is_rare & ~np.isin(hsh, th))
    yv = y[ev]
    if yv.sum() == 0 or (yv == 0).sum() == 0:
        print(fname, "skipped (no attacks or benign in test block)"); continue
    bid = pd.factorize(pd.Series(tag[ev]).astype(str) + "_" + pd.Series(row[ev] // CFG["block"]).astype(str))[0]
    F = ssm_features(held)
    print(f"\n{fname}: train {len(tr):,}, eval(test) {len(ev):,} (attacks {int(yv.sum()):,}); SSM {time.time() - t0:.0f}s", flush=True)
    get = lambda var, r: X[r] if var == "base" else np.hstack([X[r], F[r]])
    keep = {}
    for var in ["base", "ssm"]:
        imp = SimpleImputer(strategy="median").fit(get(var, tr))
        Xtr, ytr = imp.transform(get(var, tr)).astype("float32"), y[tr]
        Xthr, Xev = (imp.transform(get(var, i)).astype("float32") for i in (thr_idx, ev))
        for seed in range(args.seeds):
            clf = RandomForestClassifier(random_state=seed, **CFG["rf"]).fit(Xtr, ytr)
            thr = float(np.quantile(clf.predict_proba(Xthr)[:, 1], 1 - CFG["fpr_budget"]))
            s = clf.predict_proba(Xev)[:, 1]; pred = s > thr
            tp, fn = int((pred & (yv == 1)).sum()), int((~pred & (yv == 1)).sum())
            fp, tn = int((pred & (yv == 0)).sum()), int((~pred & (yv == 0)).sum())
            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            oth = np.quantile(s[yv == 0], 1 - CFG["fpr_budget"])
            n_inc, n_det = incident_stats(ev, yv, pred)
            rows.append(dict(fold=fname, variant=var, seed=seed, n_attack=tp + fn, n_benign=fp + tn, tp=tp, fn=fn, fp=fp, tn=tn,
                             accuracy=(tp + tn) / len(yv), precision=prec, recall=rec,
                             f1=2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0, fpr=fp / max(fp + tn, 1),
                             roc_auc=roc_auc_score(yv, s), pr_auc=average_precision_score(yv, s),
                             recall_oracle=float((s[yv == 1] > oth).mean()), incidents=n_inc, incidents_detected=n_det,
                             threshold=thr))
            for fam in np.unique(lab[ev][yv == 1]):
                mm = lab[ev] == fam
                fam_rows.append(dict(fold=fname, variant=var, seed=seed, family=fam, n=int(mm.sum()), recall=float(pred[mm].mean())))
            if seed == 0: keep[var] = (s, thr)
        print(f"  {var} done ({time.time() - t0:.0f}s)", flush=True)
    for var in ["base", "ssm"]:
        ci = boot_ci(yv, keep[var][0], keep[var][1], bid, rng, other=keep["base"] if var == "ssm" else None)
        ci_rows.append(dict(fold=fname, variant=var, **ci))
    pd.DataFrame(rows).to_csv(R / "final_metrics_per_seed.csv", index=False)
    pd.DataFrame(fam_rows).to_csv(R / "final_family_recall.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(R / "final_block_bootstrap_ci_seed0.csv", index=False)

res, ci = pd.DataFrame(rows), pd.DataFrame(ci_rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
met = ["roc_auc", "pr_auc", "recall", "fpr", "precision", "f1", "accuracy", "recall_oracle"]
summ = res.groupby(["fold", "variant"])[met].mean().round(4)
summ.to_csv(R / "final_summary.csv")
print("\n=== FINAL TEST: means over seeds (threshold = 1% FPR on validation benign) ===\n", summ.to_string())
inc = res[res.seed == 0].pivot_table(index="fold", columns="variant", values=["incidents", "incidents_detected"])
print("\n=== incident-level (seed 0): incidents / detected ===\n", inc.to_string())
print("\n=== seed-0 block-bootstrap 95% CIs (d_* = ssm minus base) ===\n", ci.round(4).to_string(index=False))
pe = res[(res.fold == "per_episode:test") & (res.seed == 0)]
if len(pe):
    print("\n=== per-episode test confusion counts (seed 0) ===\n", pe[["variant", "tp", "fn", "fp", "tn"]].to_string(index=False))
fr = pd.DataFrame(fam_rows)
print("\n=== per-family recall (mean over seeds) ===\n",
      fr.pivot_table(index=["fold", "family"], columns="variant", values="recall", aggfunc="mean").round(3).to_string())
lfo = summ.reset_index(); lfo = lfo[lfo.fold != "per_episode:test"]
print("\n=== LFO macro-average over folds ===\n", lfo.groupby("variant")[["recall", "fpr", "roc_auc", "recall_oracle"]].mean().round(4).to_string())
print("\nsaved to", R)