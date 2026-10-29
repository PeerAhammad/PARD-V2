"""Split manifest v1: per-episode chronological blocks with embargo."""
from pathlib import Path
import numpy as np, pandas as pd

P, OUT, REP = Path("data/processed"), Path("data/splits"), Path("reports")
OUT.mkdir(exist_ok=True)
MIN_FAMILY, TRAIN_FRAC, VAL_FRAC, EMB_MAX, EMB_FRAC = 100, 0.60, 0.80, 1000, 0.02

cols = ["Label", "file_tag", "file_order", "row_in_file", "feat_hash"]
df = pd.concat([pd.read_parquet(p, columns=cols) for p in P.glob("*.parquet")],
               ignore_index=True)
df = df.sort_values(["file_order", "row_in_file"]).reset_index(drop=True)
fam_n = df.loc[df.Label != "BENIGN", "Label"].value_counts()
major = sorted(fam_n[fam_n >= MIN_FAMILY].index)
rare = sorted(set(fam_n.index) - set(major))
df["episode"], df["split"] = "", ""

for tag, g in df.groupby("file_tag", sort=False):
    idx, lab, n = g.index.to_numpy(), g["Label"].to_numpy(), len(g)
    fams = sorted((np.flatnonzero(lab == f).min(), np.flatnonzero(lab == f).max(), f)
                  for f in major if (lab == f).any())
    if not fams:
        bounds = [(0, n - 1, f"{tag}|all", None)]
    else:
        bounds, start = [], 0
        for i, (a, b, f) in enumerate(fams):
            end = n - 1 if i == len(fams) - 1 else (b + fams[i + 1][0]) // 2
            bounds.append((start, end, f"{tag}|{f}", f)); start = end + 1
    for s, e, name, fam in bounds:
        m = e - s + 1
        emb = min(EMB_MAX, int(EMB_FRAC * m))
        att = np.flatnonzero(lab[s:e + 1] == fam) if fam else np.array([], dtype=int)
        if len(att) >= MIN_FAMILY:   # cut by attack-row quantiles, not raw row count
            c1 = int(att[int(TRAIN_FRAC * len(att))])
            c2 = int(att[int(VAL_FRAC * len(att))])
        else:
            c1, c2 = int(TRAIN_FRAC * m), int(VAL_FRAC * m)
        df.loc[idx[s:e + 1], "episode"] = name
        df.loc[idx[s:s + c1], "split"] = "train"
        df.loc[idx[s + c1:s + c1 + emb], "split"] = "embargo"
        df.loc[idx[s + c1 + emb:s + c2], "split"] = "val"
        df.loc[idx[s + c2:s + c2 + emb], "split"] = "embargo"
        df.loc[idx[s + c2 + emb:e + 1], "split"] = "test"

df.loc[df.Label.isin(rare), "split"] = "rare_report"

# conflict flag: same feature vector seen as both benign and attack anywhere
df["is_att"] = df.Label != "BENIGN"
g = df.groupby("feat_hash")["is_att"].agg(["min", "max"])
df["conflict"] = df["feat_hash"].isin(g.index[g["min"] != g["max"]])
# leakage flag: eval rows whose exact feature vector occurs in a training row
train_h = df.loc[df.split == "train", "feat_hash"].unique()
df["leak_in_train"] = df["feat_hash"].isin(train_h) & df["split"].isin(["val", "test", "rare_report"])

df[["file_order", "row_in_file", "file_tag", "Label", "episode", "split",
    "conflict", "leak_in_train"]].to_parquet(OUT / "splits_v1.parquet", index=False)

df["eligible"] = ~df.conflict & ~df.leak_in_train
T = df.pivot_table(index="Label", columns="split", values="row_in_file", aggfunc="size", fill_value=0)
E = (df[df.split.isin(["val", "test"]) & df.eligible]
     .pivot_table(index="Label", columns="split", values="row_in_file", aggfunc="size", fill_value=0))
txt = ["# Split manifest v1", "", "## All rows by split", "", T.to_string(), "",
       "## Eligible val/test rows (no label conflict, no exact-match with train)", "", E.to_string(), "",
       f"Rare families (report-only): {rare}"]
(REP / "split_summary_v1.md").write_text("\n".join(txt))
print("\n".join(txt))