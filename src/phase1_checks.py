import pandas as pd
from pathlib import Path

cols = ["Label", "file_tag", "file_order", "row_in_file", "feat_hash"]
df = pd.concat([pd.read_parquet(p, columns=cols)
                for p in Path("data/processed").glob("*.parquet")], ignore_index=True)
df = df.sort_values(["file_order", "row_in_file"]).reset_index(drop=True)

print("== Label conflicts ==")
d = df[df.duplicated("feat_hash", keep=False)]
lab = d.groupby("feat_hash")["Label"].agg(lambda s: " | ".join(sorted(set(s))))
print(lab[lab.str.contains(r"\|", regex=True)].value_counts().head(15))

print("\n== Attack layout in row order (per file) ==")
for tag, g in df.groupby("file_tag", sort=False):
    g = g.reset_index(drop=True)
    seg = (g["Label"] != g["Label"].shift()).cumsum()
    print(f"\n{tag}: {len(g):,} rows, {seg.nunique():,} label segments")
    a = g[g["Label"] != "BENIGN"]
    for lbl, h in a.groupby("Label"):
        first, last = h["row_in_file"].min(), h["row_in_file"].max()
        span = last - first + 1
        print(f"  {lbl:28s} n={len(h):7,}  rows {first:,}-{last:,}  "
              f"({first/len(g):.0%}-{last/len(g):.0%} of file)  density={len(h)/span:.2f}")