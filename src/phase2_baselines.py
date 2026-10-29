"""Phase 2 baselines: majority class, Random Forest, Isolation Forest on frozen splits_v1."""
import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd, sklearn
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--max-benign-train", type=int, default=600_000)
ap.add_argument("--eval-test", action="store_true")
ap.add_argument("--tag", default="")
args = ap.parse_args()

CFG = dict(splits="splits_v1", seeds=list(range(args.seeds)), fpr_budget=0.01,
           max_benign_train=args.max_benign_train, eval_test=args.eval_test,
           rf=dict(n_estimators=200, max_samples=0.25, min_samples_leaf=2, n_jobs=-1),
           iforest=dict(n_estimators=200, max_samples=256, n_jobs=-1, fit_rows=200_000))

rid = time.strftime("baselines_%Y%m%d_%H%M%S") + (f"_{args.tag}" if args.tag else "")
R = Path("results") / rid; R.mkdir(parents=True)
(R / "config.json").write_text(json.dumps(CFG, indent=2))
(R / "env.json").write_text(json.dumps(dict(python=sys.version, platform=platform.platform(),
    sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__), indent=2))

# ---- load
D = pd.concat([pd.read_parquet(p) for p in sorted(Path("data/processed").glob("*.parquet"))], ignore_index=True)
S = pd.read_parquet("data/splits/splits_v1.parquet")[["file_order", "row_in_file", "split", "conflict", "leak_in_train"]]
D = D.merge(S, on=["file_order", "row_in_file"], how="left", validate="one_to_one")
assert D["split"].notna().all()
META = {"Label", "file_tag", "file_order", "row_in_file", "feat_hash", "dup_in_file",
        "has_invalid", "split", "conflict", "leak_in_train"}
FEATS = [c for c in D.columns if c not in META]
D["y"] = (D["Label"] != "BENIGN").astype(int)
print("features:", len(FEATS))

# ---- training set: no conflicts, deduplicated, benign capped (fixed subsample)
tr = D[(D.split == "train") & ~D.conflict].drop_duplicates("feat_hash")
ben = tr[tr.y == 0]
if len(ben) > CFG["max_benign_train"]:
    ben = ben.sample(CFG["max_benign_train"], random_state=0)
tr = pd.concat([tr[tr.y == 1], ben])
imp = SimpleImputer(strategy="median").fit(tr[FEATS])          # fitted on train only
Xtr, ytr = imp.transform(tr[FEATS]).astype("float32"), tr["y"].to_numpy()
print("train rows:", len(tr), "attacks:", int(ytr.sum()))

# ---- evaluation rows (test is only touched with --eval-test)
splits_eval = ["val"] + (["test", "rare_report"] if args.eval_test else [])
E = D[D.split.isin(splits_eval) & ~D.conflict].reset_index(drop=True)
XE = imp.transform(E[FEATS]).astype("float32")
yE, fam = E["y"].to_numpy(), E["Label"].to_numpy()
spl, leak = E["split"].to_numpy(), E["leak_in_train"].to_numpy()
if args.eval_test:
    with open("results/test_access_log.csv", "a") as f:
        f.write(f"{time.ctime()},{rid}\n")

def build(name, seed):
    if name == "majority":
        return lambda X: np.zeros(len(X))
    if name == "rf":
        clf = RandomForestClassifier(random_state=seed, **CFG["rf"]).fit(Xtr, ytr)
        return lambda X: clf.predict_proba(X)[:, 1]
    kw = {k: v for k, v in CFG["iforest"].items() if k != "fit_rows"}
    b = np.flatnonzero(ytr == 0)
    b = np.random.RandomState(seed).choice(b, min(CFG["iforest"]["fit_rows"], len(b)), replace=False)
    iso = IsolationForest(random_state=seed, **kw).fit(Xtr[b])
    return lambda X: -iso.score_samples(X)

def evaluate(y, s, f, thr):
    pred = s > thr
    out = {"n_attack": int(y.sum()), "n_benign": int((y == 0).sum()),
           "recall": float(pred[y == 1].mean()),
           "family_recall": {k: float(pred[(f == k) & (y == 1)].mean()) for k in np.unique(f[y == 1])}}
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    out.update(tp=tp, fn=fn)
    if out["n_benign"] > 0:
        fp, tn = int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum())
        prec = tp / max(tp + fp, 1)
        out.update(fp=fp, tn=tn, accuracy=(tp + tn) / len(y))
        out.update(fpr=fp / out["n_benign"], precision=prec,
                   f1=2 * prec * out["recall"] / (prec + out["recall"]) if prec + out["recall"] > 0 else 0.0,
                   roc_auc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)),
                   prevalence=float(y.mean()))
    return out

rows, fam_rows, SC, THR = [], [], {}, {}
for model in ["majority", "rf", "iforest"]:
    for seed in ([0] if model == "majority" else CFG["seeds"]):
        t0 = time.time()
        s = build(model, seed)(XE)
        SC[f"{model}_{seed}"] = s.astype("float32")
        vm = (spl == "val") & ~leak & (yE == 0)
        thr = np.quantile(s[vm], 1 - CFG["fpr_budget"])        # threshold from validation only
        THR[f"{model}_{seed}"] = float(thr)
        for setname in ["filtered", "nonconflict"]:
            for sp in splits_eval:
                m = spl == sp
                if setname == "filtered" and sp != "rare_report":
                    m = m & ~leak
                if setname == "nonconflict" and sp == "rare_report":
                    continue
                if m.sum() == 0:
                    continue
                r = evaluate(yE[m], s[m], fam[m], thr)
                for k, v in r.pop("family_recall").items():
                    fam_rows.append(dict(model=model, set=setname, split=sp, seed=seed, family=k, recall=v))
                rows.append(dict(model=model, set=setname, split=sp, seed=seed, threshold=float(thr), **r))
        print(f"{model} seed {seed} done in {time.time() - t0:.0f}s")

res, fr = pd.DataFrame(rows), pd.DataFrame(fam_rows)
np.savez_compressed(R / "scores.npz", y=yE, fam=fam.astype(str), split=spl.astype(str), leak=leak,
                    file_tag=E["file_tag"].to_numpy().astype(str), row=E["row_in_file"].to_numpy(), **SC)
(R / "thresholds.json").write_text(json.dumps(THR, indent=2))
res.to_csv(R / "metrics_per_seed.csv", index=False)
fr.to_csv(R / "family_recall_per_seed.csv", index=False)
met = ["roc_auc", "pr_auc", "recall", "precision", "f1", "fpr"]
summ = res.groupby(["model", "set", "split"])[[c for c in met if c in res]].agg(["mean", "std"]).round(4)
summ.to_csv(R / "metrics_summary.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("\n", summ.to_string())
fp = fr[fr.set == "filtered"].pivot_table(index="family", columns=["model", "split"], values="recall", aggfunc="mean").round(3)
print("\nPer-family recall (filtered set, threshold at 1% val FPR):\n", fp.to_string())
print("\nsaved to", R)