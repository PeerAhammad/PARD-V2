"""Phase 2b: leave-family-out and leave-day-out evaluation on frozen splits_v1."""
import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd, sklearn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--only", default="", help="comma-separated substrings of fold names")
ap.add_argument("--eval-test", action="store_true")
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--boot-auc", type=int, default=100)
ap.add_argument("--tag", default="")
args = ap.parse_args()

CFG = dict(fpr_budget=0.01, max_benign_train=600_000, block=5000, if_fit_rows=200_000,
           rf=dict(n_estimators=200, max_samples=0.25, min_samples_leaf=2, n_jobs=-1),
           seeds=args.seeds, eval_test=args.eval_test, only=args.only)
rid = time.strftime("lfo_%Y%m%d_%H%M%S") + (f"_{args.tag}" if args.tag else "")
R = Path("results") / rid; R.mkdir(parents=True)
(R / "config.json").write_text(json.dumps(CFG, indent=2))
(R / "env.json").write_text(json.dumps(dict(python=sys.version, platform=platform.platform(),
    sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__), indent=2))
if args.eval_test:
    with open("results/test_access_log.csv", "a") as f:
        f.write(f"{time.ctime()},{rid}\n")

# ---- load
D = pd.concat([pd.read_parquet(p) for p in sorted(Path("data/processed").glob("*.parquet"))], ignore_index=True)
S = pd.read_parquet("data/splits/splits_v1.parquet")[["file_order", "row_in_file", "episode", "split", "conflict"]]
D = D.merge(S, on=["file_order", "row_in_file"], how="left", validate="one_to_one")
assert D["split"].notna().all()
META = {"Label", "file_tag", "file_order", "row_in_file", "feat_hash", "dup_in_file",
        "has_invalid", "split", "conflict", "episode"}
FEATS = [c for c in D.columns if c not in META]
X = D[FEATS].to_numpy(dtype="float32")
lab = D["Label"].to_numpy(); y = (lab != "BENIGN").astype(int)
spl, conf, ep = D["split"].to_numpy(), D["conflict"].to_numpy(), D["episode"].to_numpy()
tag, row, hsh = D["file_tag"].to_numpy(), D["row_in_file"].to_numpy(), D["feat_hash"].to_numpy()
del D
RARE = ["Heartbleed", "Infiltration", "Web Attack - Sql Injection"]
is_rare = np.isin(lab, RARE)

# ---- folds
major = sorted(set(lab[y == 1]) - set(RARE))
folds = []
for f in major:
    e = np.unique(ep[lab == f]); assert len(e) == 1, (f, e)
    folds.append((f"family:{f}", "family", ep == e[0]))
for t in ["tue", "wed", "thu_am_web", "fri_am", "fri_pm_scan", "fri_pm_ddos"]:
    folds.append((f"day:{t}", "day", tag == t))
if args.only:
    keys = [k.strip() for k in args.only.split(",")]
    folds = [f for f in folds if any(k in f[0] for k in keys)]
print("folds:", [f[0] for f in folds])

def fit_score(model, seed, Xtr, ytr):
    if model == "rf":
        clf = RandomForestClassifier(random_state=seed, **CFG["rf"]).fit(Xtr, ytr)
        return lambda Z: clf.predict_proba(Z)[:, 1]
    b = np.flatnonzero(ytr == 0)
    b = np.random.RandomState(seed).choice(b, min(CFG["if_fit_rows"], len(b)), replace=False)
    iso = IsolationForest(random_state=seed, n_estimators=200, max_samples=256, n_jobs=-1).fit(Xtr[b])
    return lambda Z: -iso.score_samples(Z)

def block_boot(yv, sv, thr, bid, rng):
    o = np.argsort(bid, kind="stable"); yv, sv, bid = yv[o], sv[o], bid[o]
    _, st = np.unique(bid, return_index=True); en = np.append(st[1:], len(yv)); nb = len(st)
    pred = sv > thr
    cnt = lambda m: np.add.reduceat(m.astype(np.int64), st)
    tp, fn = cnt(pred & (yv == 1)), cnt(~pred & (yv == 1))
    fp, tn = cnt(pred & (yv == 0)), cnt(~pred & (yv == 0))
    dr = rng.integers(0, nb, size=(args.boot, nb))
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = tp[dr].sum(1) / (tp[dr].sum(1) + fn[dr].sum(1))
        fpr = fp[dr].sum(1) / (fp[dr].sum(1) + tn[dr].sum(1))
    aucs = []
    for _ in range(args.boot_auc):
        d = rng.integers(0, nb, nb)
        idx = np.concatenate([np.arange(st[i], en[i]) for i in d])
        if len(np.unique(yv[idx])) == 2:
            aucs.append(roc_auc_score(yv[idx], sv[idx]))
    q = lambda v: tuple(np.nanpercentile(v, [2.5, 97.5])) if len(v) else (np.nan, np.nan)
    return dict(zip(["recall_lo", "recall_hi", "fpr_lo", "fpr_hi", "auc_lo", "auc_hi"],
                    [*q(rec), *q(fpr), *q(np.array(aucs))]))

rows, fam_rows = [], []
rng = np.random.default_rng(0)
evsets = {"dev": ["train", "embargo", "val"]}
if args.eval_test:
    evsets["test"] = ["test"]

for fname, kind, held in folds:
    t0 = time.time()
    tr_all = np.flatnonzero((spl == "train") & ~conf & ~held)
    train_hashes = np.unique(hsh[tr_all])
    tr = tr_all[~pd.Series(hsh[tr_all]).duplicated().to_numpy()]
    atk, ben = tr[y[tr] == 1], tr[y[tr] == 0]
    if len(ben) > CFG["max_benign_train"]:
        ben = np.random.RandomState(0).choice(ben, CFG["max_benign_train"], replace=False)
    tr = np.concatenate([atk, ben])
    imp = SimpleImputer(strategy="median").fit(X[tr])           # fitted on fold-train only
    Xtr, ytr = imp.transform(X[tr]).astype("float32"), y[tr]
    thr_idx = np.flatnonzero((spl == "val") & ~held & ~conf & (y == 0) & ~np.isin(hsh, train_hashes))
    Xthr = imp.transform(X[thr_idx]).astype("float32")
    EV = {}
    for en, sp in evsets.items():
        idx = np.flatnonzero(held & np.isin(spl, sp) & ~conf & ~is_rare & ~np.isin(hsh, train_hashes))
        if len(idx): EV[en] = (idx, imp.transform(X[idx]).astype("float32"))
    print(f"\n{fname}: train {len(tr):,} (attacks {len(atk):,}), threshold rows {len(thr_idx):,}, "
          f"eval { {k: len(v[0]) for k, v in EV.items()} }")
    for model in ["rf", "iforest"]:
        for seed in range(args.seeds):
            sc = fit_score(model, seed, Xtr, ytr)
            thr = float(np.quantile(sc(Xthr), 1 - CFG["fpr_budget"]))
            for en, (idx, Xe) in EV.items():
                s, yv = sc(Xe), y[idx]
                na, nbn = int(yv.sum()), int((yv == 0).sum())
                if na == 0 or nbn == 0:
                    continue
                pred = s > thr
                oth = np.quantile(s[yv == 0], 1 - CFG["fpr_budget"])
                r = dict(fold=fname, kind=kind, model=model, seed=seed, evalset=en, n_attack=na, n_benign=nbn,
                         n_train=len(tr), roc_auc=roc_auc_score(yv, s), pr_auc=average_precision_score(yv, s),
                         threshold=thr, recall_valthr=float(pred[yv == 1].mean()),
                         fpr_valthr=float(pred[yv == 0].mean()), recall_oracle=float((s[yv == 1] > oth).mean()))
                if seed == 0:
                    bid = pd.factorize(pd.Series(tag[idx]).astype(str) + "_" + pd.Series(row[idx] // CFG["block"]).astype(str))[0]
                    r.update(block_boot(yv, s, thr, bid, rng))
                rows.append(r)
                for f in np.unique(lab[idx][yv == 1]):
                    mm = (lab[idx] == f)
                    fam_rows.append(dict(fold=fname, model=model, seed=seed, evalset=en, family=f,
                                         n=int(mm.sum()), recall=float(pred[mm].mean())))
    print(f"  fold done in {time.time() - t0:.0f}s")
    pd.DataFrame(rows).to_csv(R / "lfo_metrics_per_seed.csv", index=False)   # incremental save

res, fr = pd.DataFrame(rows), pd.DataFrame(fam_rows)
res.to_csv(R / "lfo_metrics_per_seed.csv", index=False)
fr.to_csv(R / "lfo_family_recall_per_seed.csv", index=False)
met = ["roc_auc", "pr_auc", "recall_valthr", "fpr_valthr", "recall_oracle"]
summ = res.groupby(["evalset", "model", "fold"])[met].agg(["mean", "std"]).round(3)
summ.to_csv(R / "lfo_summary.csv")
ci = res[res.seed == 0][["evalset", "model", "fold", "recall_valthr", "recall_lo", "recall_hi",
                         "fpr_valthr", "fpr_lo", "fpr_hi", "roc_auc", "auc_lo", "auc_hi"]].round(3)
ci.to_csv(R / "lfo_seed0_block_bootstrap_ci.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
mean = res.groupby(["evalset", "model", "fold"])[met].mean().round(3)
print("\n=== means over seeds ===\n", mean.to_string())
print("\n=== seed 0 with block-bootstrap 95% CIs ===\n", ci.to_string(index=False))

# ---- figure: RF recall (val-threshold vs oracle) and achieved FPR, dev set
for en in res.evalset.unique():
    d = res[(res.model == "rf") & (res.evalset == en)].groupby("fold")[["recall_valthr", "recall_oracle", "fpr_valthr"]].mean()
    if d.empty: continue
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6)); x = np.arange(len(d)); w = 0.4
    ax[0].bar(x - w / 2, d.recall_valthr, w, label="threshold from other episodes' val benign")
    ax[0].bar(x + w / 2, d.recall_oracle, w, label="oracle 1% FPR on held-out benign")
    ax[0].set_ylabel("Recall"); ax[0].set_title(f"RF held-out recall ({en})")
    ax[1].bar(x, d.fpr_valthr, color="tab:red"); ax[1].axhline(0.01, c="k", ls="--", label="1% target")
    ax[1].set_yscale("log"); ax[1].set_ylabel("FPR on held-out benign"); ax[1].set_title("Achieved FPR at transferred threshold")
    for a in ax:
        a.set_xticks(x, d.index, rotation=50, ha="right", fontsize=8); a.legend(fontsize=7); a.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(R / f"lfo_rf_{en}.png", dpi=150); plt.close(fig)
print("\nsaved to", R)