"""Build demo\\test_upload.csv for the 'Score your own CSV' page from HELD-OUT test-block rows only.
A mix of benign flows and every major attack family, so the demo shows both detections and misses."""
from pathlib import Path
import numpy as np, pandas as pd

PER_FAMILY, N_BENIGN = 300, 4000
cols = None
D = pd.concat([pd.read_parquet(p) for p in sorted(Path("data/processed").glob("*.parquet"))], ignore_index=True)
S = pd.read_parquet("data/splits/splits_v1.parquet")[["file_order", "row_in_file", "split", "conflict", "leak_in_train"]]
D = D.merge(S, on=["file_order", "row_in_file"], how="left", validate="one_to_one")
D = D[(D.split == "test") & ~D.conflict & ~D.leak_in_train]
META = {"Label", "file_tag", "file_order", "row_in_file", "feat_hash", "dup_in_file", "has_invalid", "split", "conflict", "leak_in_train"}
feats = [c for c in D.columns if c not in META]
rng = np.random.RandomState(7)
parts = [D[D.Label == "BENIGN"].sample(min(N_BENIGN, int((D.Label == "BENIGN").sum())), random_state=7)]
for fam in sorted(set(D.Label) - {"BENIGN"}):
    g = D[D.Label == fam]; parts.append(g.sample(min(PER_FAMILY, len(g)), random_state=7))
out = pd.concat(parts).sample(frac=1, random_state=7)
csv = out[feats].copy()
csv.columns = [" " + c for c in csv.columns]        # raw CICIDS files have leading spaces in column names
csv["Label"] = out["Label"].values
Path("demo").mkdir(exist_ok=True); csv.to_csv("demo/test_upload.csv", index=False)
print(f"wrote demo/test_upload.csv: {len(csv):,} flows, {len(feats)} features")
print(csv["Label"].value_counts().to_string())