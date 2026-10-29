"""Build the demo assets. Trains the final base RF with the frozen protocol (same data, hyper-parameters and
seed 0 as the per-episode base model in the evaluation), fixes the 1%-FPR threshold on validation benign rows,
and pre-scores every val/test/rare row so the website needs no model at replay time."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

OUT = Path("demo"); OUT.mkdir(exist_ok=True)
t0 = time.time()
D = pd.concat([pd.read_parquet(p) for p in sorted(Path("data/processed").glob("*.parquet"))], ignore_index=True)
S = pd.read_parquet("data/splits/splits_v1.parquet")[["file_order", "row_in_file", "episode", "split", "conflict", "leak_in_train"]]
D = D.merge(S, on=["file_order", "row_in_file"], how="left", validate="one_to_one")
assert D["split"].notna().all()
META = {"Label", "file_tag", "file_order", "row_in_file", "feat_hash", "dup_in_file", "has_invalid",
        "split", "conflict", "episode", "leak_in_train"}
FEATS = [c for c in D.columns if c not in META]
X = D[FEATS].to_numpy(dtype="float32")
lab = D["Label"].to_numpy(); y = (lab != "BENIGN").astype(int)
spl, conf, leak, hsh = D["split"].to_numpy(), D["conflict"].to_numpy(), D["leak_in_train"].to_numpy(), D["feat_hash"].to_numpy()

tr = np.flatnonzero((spl == "train") & ~conf)
tr = tr[~pd.Series(hsh[tr]).duplicated().to_numpy()]
atk, ben = tr[y[tr] == 1], tr[y[tr] == 0]
if len(ben) > 600_000:
    ben = np.random.RandomState(0).choice(ben, 600_000, replace=False)
tr = np.concatenate([atk, ben])
imp = SimpleImputer(strategy="median").fit(X[tr])
clf = RandomForestClassifier(n_estimators=200, max_samples=0.25, min_samples_leaf=2, n_jobs=-1, random_state=0)
clf.fit(imp.transform(X[tr]).astype("float32"), y[tr])
print(f"trained on {len(tr):,} rows ({time.time() - t0:.0f}s)")

def score(rows, chunk=200_000):
    return np.concatenate([clf.predict_proba(imp.transform(X[rows[i:i + chunk]]).astype("float32"))[:, 1]
                           for i in range(0, len(rows), chunk)]) if len(rows) else np.array([])

vb = np.flatnonzero((spl == "val") & ~conf & ~leak & (y == 0))
thr = float(np.quantile(score(vb), 0.99))
rows = np.flatnonzero(np.isin(spl, ["val", "test", "rare_report"]))
sc = score(rows)
out = pd.DataFrame({"file_tag": D["file_tag"].to_numpy()[rows], "row_in_file": D["row_in_file"].to_numpy()[rows],
                    "Label": lab[rows], "episode": D["episode"].to_numpy()[rows], "split": spl[rows],
                    "conflict": conf[rows], "leak_in_train": leak[rows], "score": sc.astype("float32")})
out["alert"] = out["score"] > thr
out.to_parquet(OUT / "scores.parquet", index=False)
joblib.dump(dict(imputer=imp, clf=clf, feats=FEATS, threshold=thr), OUT / "model.joblib", compress=3)

t = out[(out.split == "test") & ~out.conflict & ~out.leak_in_train]
ya = (t.Label != "BENIGN").to_numpy(); al = t.alert.to_numpy()
tp, fn, fp, tn = int((al & ya).sum()), int((~al & ya).sum()), int((al & ~ya).sum()), int((~al & ~ya).sum())
meta = dict(threshold=thr, n_train=int(len(tr)), n_features=len(FEATS), test_eval=dict(
    tp=tp, fn=fn, fp=fp, tn=tn, recall=tp / (tp + fn), fpr=fp / (fp + tn), accuracy=(tp + tn) / len(t),
    precision=tp / max(tp + fp, 1)))
(OUT / "meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta["test_eval"], indent=2)); print(f"done in {time.time() - t0:.0f}s; assets in {OUT}")