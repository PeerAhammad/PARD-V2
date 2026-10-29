"""Phase 3: causal temporal features vs base features on frozen LFO dev folds (+ per-episode control).
No test blocks are used anywhere in this script."""
import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd, sklearn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--variants", default="base,temporal,shuffled")
ap.add_argument("--only", default="")
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--boot-auc", type=int, default=100)
ap.add_argument("--lfo-ref", default="", help="path to lfo_metrics_per_seed.csv for a base reproduction check")
ap.add_argument("--tag", default="")
args = ap.parse_args()

CFG = dict(fpr_budget=0.01, max_benign_train=600_000, block=5000, windows=[50, 200, 1000],
           port_windows=[200, 1000], rf=dict(n_estimators=200, max_samples=0.25, min_samples_leaf=2, n_jobs=-1),
           seeds=args.seeds, variants=args.variants, only=args.only)
rid = time.strftime("phase3_%Y%m%d_%H%M%S") + (f"_{args.tag}" if args.tag else "")
R = Path("results") / rid; R.mkdir(parents=True)
(R / "config.json").write_text(json.dumps(CFG, indent=2))
(R / "env.json").write_text(json.dumps(dict(python=sys.version, platform=platform.platform(),
    sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__), indent=2))

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
NEED = CORE + ["SYN Flag Count", "ACK Flag Count", "Destination Port"]
miss = [c for c in NEED if c not in fi]
assert not miss, f"missing columns {miss}; available: {FEATS}"
WINS, PW = CFG["windows"], CFG["port_windows"]
GCOLS = {"diff": [f"diff|{c}" for c in CORE],
         "roll": [f"{s}{w}|{c}" for w in WINS for s in ("mean", "std") for c in CORE],
         "ind": [f"{n}{w}" for n in ("small", "synnoack", "noreply") for w in WINS],
         "port": [f"ports{w}" for w in PW]}
VARIANTS = {"base": ([], False),
            "temporal": (["diff", "roll", "ind", "port"], False),
            "shuffled": (["diff", "roll", "ind", "port"], True),
            "diff": (["diff"], False), "roll": (["roll"], False), "indport": (["ind", "port"], False)}
variants = [v.strip() for v in args.variants.split(",")]
assert "base" in variants and all(v in VARIANTS for v in variants)

def distinct_frac(ports, W):
    out = np.empty(len(ports)); cnt = {}
    for i, p in enumerate(ports):
        cnt[p] = cnt.get(p, 0) + 1
        if i >= W:
            q = ports[i - W]; c = cnt[q] - 1
            if c: cnt[q] = c
            else: del cnt[q]
        out[i] = len(cnt)
    return out / np.minimum(np.arange(1, len(ports) + 1), W)

def build_groups(shuffle):
    rng = np.random.default_rng(123)
    out = {g: np.zeros((N, len(c)), "float32") for g, c in GCOLS.items()}
    ci = [fi[c] for c in CORE]
    for t in np.unique(tag):
        idx = np.flatnonzero(tag == t)
        idx = idx[np.argsort(row[idx], kind="stable")]
        seq = rng.permutation(idx) if shuffle else idx          # shuffle = null control
        core = pd.DataFrame(X[seq][:, ci].astype("float64"))
        out["diff"][seq] = core.diff().fillna(0).to_numpy()
        parts = []
        for w in WINS:
            r = core.rolling(w, min_periods=1)
            parts += [r.mean().to_numpy(), r.std().fillna(0).to_numpy()]
        out["roll"][seq] = np.hstack(parts)
        fwd, bwd = X[seq, fi["Total Fwd Packets"]], X[seq, fi["Total Backward Packets"]]
        syn, ack = X[seq, fi["SYN Flag Count"]], X[seq, fi["ACK Flag Count"]]
        inds = [((fwd + bwd) <= 2), ((syn > 0) & (ack == 0)), (bwd == 0)]
        out["ind"][seq] = np.column_stack([pd.Series(v.astype("float64")).rolling(w, min_periods=1).mean().to_numpy()
                                            for v in inds for w in WINS])
        ports = np.nan_to_num(X[seq, fi["Destination Port"]], nan=-1).astype(np.int64).tolist()
        out["port"][seq] = np.column_stack([distinct_frac(ports, w) for w in PW])
        print(f"  features built for {t} (shuffle={shuffle})", flush=True)
    return out

G = {}
if any(not VARIANTS[v][1] and VARIANTS[v][0] for v in variants):
    print("building causal features..."); G[False] = build_groups(False)
if any(VARIANTS[v][1] for v in variants):
    print("building shuffled-order features..."); G[True] = build_groups(True)

def get(var, rows):
    groups, shuf = VARIANTS[var]
    return np.hstack([X[rows]] + [G[shuf][g][rows] for g in groups])

# ---- folds (14 distinct LFO folds + per-episode control)
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

def fit_rf(seed, Xtr, ytr):
    clf = RandomForestClassifier(random_state=seed, **CFG["rf"]).fit(Xtr, ytr)
    return lambda Z: clf.predict_proba(Z)[:, 1]

def paired_boot(yv, sa, ta, sb, tb, bid, rng):
    """a = base, b = variant; returns CIs of (variant - base)."""
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

rows, delta_rows = [], []
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
        print(fname, "skipped (no attacks or no benign in eval set)"); continue
    bid = pd.factorize(pd.Series(tag[ev]).astype(str) + "_" + pd.Series(row[ev] // CFG["block"]).astype(str))[0]
    print(f"\n{fname}: train {len(tr):,}, thr rows {len(thr_idx):,}, eval {len(ev):,} (attacks {int(yv.sum()):,})", flush=True)
    keep = {}
    for var in variants:
        imp = SimpleImputer(strategy="median").fit(get(var, tr))          # fold-train only
        Xtr, ytr = imp.transform(get(var, tr)).astype("float32"), y[tr]
        Xthr, Xev = (imp.transform(get(var, i)).astype("float32") for i in (thr_idx, ev))
        for seed in range(args.seeds):
            sc = fit_rf(seed, Xtr, ytr)
            thr = float(np.quantile(sc(Xthr), 1 - CFG["fpr_budget"])); s = sc(Xev); pred = s > thr
            oth = np.quantile(s[yv == 0], 1 - CFG["fpr_budget"])
            rows.append(dict(fold=fname, variant=var, seed=seed, n_attack=int(yv.sum()), n_benign=int((yv == 0).sum()),
                             roc_auc=roc_auc_score(yv, s), pr_auc=average_precision_score(yv, s),
                             recall_valthr=float(pred[yv == 1].mean()), fpr_valthr=float(pred[yv == 0].mean()),
                             recall_oracle=float((s[yv == 1] > oth).mean()), threshold=thr))
            if seed == 0: keep[var] = (s, thr)
        print(f"  {var} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); d = df[df.fold == fname]
    for var in variants:
        if var == "base": continue
        b = d[d.variant == "base"].set_index("seed"); v = d[d.variant == var].set_index("seed")
        r = dict(fold=fname, variant=var,
                 d_recall=(v.recall_valthr - b.recall_valthr).mean(), d_fpr=(v.fpr_valthr - b.fpr_valthr).mean(),
                 d_recall_oracle=(v.recall_oracle - b.recall_oracle).mean(), d_auc=(v.roc_auc - b.roc_auc).mean(),
                 base_recall=b.recall_valthr.mean(), var_recall=v.recall_valthr.mean(),
                 base_fpr=b.fpr_valthr.mean(), var_fpr=v.fpr_valthr.mean())
        r.update(paired_boot(yv, keep["base"][0], keep["base"][1], keep[var][0], keep[var][1], bid, rng))
        r["verdict"] = verdict(r); delta_rows.append(r)
    df.to_csv(R / "phase3_metrics_per_seed.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(R / "phase3_deltas.csv", index=False)

res, dl = pd.DataFrame(rows), pd.DataFrame(delta_rows)
res.to_csv(R / "phase3_metrics_per_seed.csv", index=False); dl.to_csv(R / "phase3_deltas.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
mean = res.groupby(["fold", "variant"])[["roc_auc", "recall_valthr", "fpr_valthr", "recall_oracle"]].mean().round(3)
print("\n=== means over seeds ===\n", mean.to_string())
cols = ["fold", "variant", "base_recall", "var_recall", "d_recall", "d_rec_lo", "d_rec_hi", "base_fpr", "var_fpr",
        "d_fpr", "d_auc", "verdict"]
print("\n=== variant minus base (CIs are seed 0, paired block bootstrap) ===\n", dl[cols].round(3).to_string(index=False))
lf = dl[dl.fold != "per_episode:val"]
print("\n=== verdict counts on the LFO folds ===\n", lf.groupby(["variant", "verdict"]).size().unstack(fill_value=0).to_string())
print("\nmean d_recall over LFO folds:", lf.groupby("variant")["d_recall"].mean().round(4).to_dict())

if args.lfo_ref and Path(args.lfo_ref).exists():
    ref = pd.read_csv(args.lfo_ref); ref = ref[(ref.model == "rf") & (ref.evalset == "dev")]
    m = res[res.variant == "base"].merge(ref, on=["fold", "seed"], suffixes=("", "_ref"))
    if len(m):
        print("\nbase reproduction check vs LFO run: max |delta recall| =",
              float((m.recall_valthr - m.recall_valthr_ref).abs().max()),
              ", max |delta AUC| =", float((m.roc_auc - m.roc_auc_ref).abs().max()))

for var in [v for v in variants if v != "base"]:
    d = dl[dl.variant == var]
    if d.empty: continue
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6)); x = np.arange(len(d))
    ax[0].bar(x, d.d_recall, yerr=[(d.d_recall - d.d_rec_lo).clip(lower=0), (d.d_rec_hi - d.d_recall).clip(lower=0)], capsize=2)
    ax[0].set_title(f"{var} minus base: recall at transferred threshold"); ax[0].set_ylabel("delta recall")
    ax[1].bar(x, d.d_fpr, color="tab:red"); ax[1].axhline(0.005, c="k", ls="--", lw=.8)
    ax[1].set_title("delta FPR on held-out benign (dashed = +0.005 limit)"); ax[1].set_ylabel("delta FPR")
    for a in ax:
        a.axhline(0, c="gray", lw=.6); a.set_xticks(x, d.fold, rotation=55, ha="right", fontsize=7); a.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(R / f"phase3_delta_{var}.png", dpi=150); plt.close(fig)
print("\nsaved to", R)