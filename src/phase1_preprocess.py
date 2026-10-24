"""Phase 1 preprocessing. Raw CSVs are never modified.
No statistics are fitted here: imputation/scaling happen later, on training data only."""
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

RAW, OUT, REP = Path("data"), Path("data/processed"), Path("reports")
OUT.mkdir(parents=True, exist_ok=True); REP.mkdir(exist_ok=True)

# (tag, filename) in chronological day order. Row order within a file is the only
# time proxy available in this release (no Timestamp column).
FILES = [
    ("mon",        "Monday-WorkingHours.pcap_ISCX.csv"),
    ("tue",        "Tuesday-WorkingHours.pcap_ISCX.csv"),
    ("wed",        "Wednesday-workingHours.pcap_ISCX.csv"),
    ("thu_am_web", "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"),
    ("thu_pm_inf", "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"),
    ("fri_am",     "Friday-WorkingHours-Morning.pcap_ISCX.csv"),
    ("fri_pm_scan","Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
    ("fri_pm_ddos","Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
]
CONSTANT = ["Bwd PSH Flags", "Bwd URG Flags", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
            "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"]
NONNEG = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s"]   # negatives are invalid

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def fix_label(s):
    s = s.str.strip()
    return s.str.replace(r"^Web Attack.*?(Brute Force|XSS|Sql Injection)$",
                         r"Web Attack - \1", regex=True)

def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None

manifest = {"script": "phase1_preprocess.py", "git_commit": git_commit(),
            "python": sys.version, "pandas": pd.__version__, "numpy": np.__version__,
            "time_proxy": "row order within file (no Timestamp in this release)",
            "dropped_columns": ["Fwd Header Length.1 (verified identical)"] + CONSTANT,
            "files": {}}
hash_frames = []

for order, (tag, fname) in enumerate(FILES):
    p = RAW / fname
    df = pd.read_csv(p, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip()
    assert (df["Fwd Header Length"] == df["Fwd Header Length.1"]).all(), fname
    df = df.drop(columns=["Fwd Header Length.1"])
    label = fix_label(df["Label"])
    feats = [c for c in df.columns if c != "Label"]
    X = df[feats].apply(pd.to_numeric, errors="coerce").astype("float64")

    # exact-duplicate hash on the raw numeric values (before any cleaning)
    h = pd.util.hash_pandas_object(X, index=False).to_numpy()

    n_inf = int(np.isinf(X.to_numpy()).sum())
    X = X.replace([np.inf, -np.inf], np.nan)
    n_neg = {c: int((X[c] < 0).sum()) for c in NONNEG}
    for c in NONNEG:
        X.loc[X[c] < 0, c] = np.nan
    for c in CONSTANT:
        assert X[c].nunique(dropna=True) <= 1, (fname, c)
    X = X.drop(columns=CONSTANT)

    out = X.copy()
    out["Label"] = label.to_numpy()
    out["file_tag"] = tag
    out["file_order"] = order
    out["row_in_file"] = np.arange(len(out))
    out["feat_hash"] = h
    out["dup_in_file"] = pd.Series(h).duplicated(keep="first").to_numpy()
    out["has_invalid"] = X.isna().any(axis=1).to_numpy()
    out_path = OUT / f"{tag}.parquet"
    out.to_parquet(out_path, index=False)

    hash_frames.append(pd.DataFrame({"file_order": order, "feat_hash": h, "Label": label.to_numpy()}))
    manifest["files"][tag] = {
        "raw_file": fname, "raw_sha256": sha256(p), "parquet_sha256": sha256(out_path),
        "rows": int(len(out)), "n_features": int(X.shape[1]),
        "labels": label.value_counts().to_dict(),
        "nan_cells_after_cleaning": int(X.isna().sum().sum()),
        "rows_with_invalid": int(out["has_invalid"].sum()),
        "inf_cells": n_inf, "negative_cells": n_neg,
        "dup_in_file": int(out["dup_in_file"].sum()),
        "dup_in_file_by_label": out.loc[out["dup_in_file"], "Label"].value_counts().to_dict(),
    }
    print(tag, len(out), "done")

H = pd.concat(hash_frames, ignore_index=True)
g = H.groupby("feat_hash")
manifest["cross_file_duplicate_hashes"] = int((g["file_order"].nunique() > 1).sum())
manifest["label_conflict_hashes"] = int((g["Label"].nunique() > 1).sum())
manifest["rows_in_label_conflict_hashes"] = int(H["feat_hash"].isin(
    g["Label"].nunique()[lambda s: s > 1].index).sum())
manifest["total_rows"] = int(len(H))
manifest["label_distribution"] = H["Label"].value_counts().to_dict()

(REP / "preprocess_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

# human-readable data-quality report
L = ["# CICIDS2017 data-quality report (Phase 1)", "",
     f"Total rows: {manifest['total_rows']:,}. Features kept: 69. "
     "Time proxy: row order within file (no Timestamp column).", "",
     "| file | rows | in-file dups | rows w/ invalid | inf cells | neg cells |", "|---|---|---|---|---|---|"]
for t, v in manifest["files"].items():
    L.append(f"| {t} | {v['rows']:,} | {v['dup_in_file']:,} | {v['rows_with_invalid']:,} | "
             f"{v['inf_cells']:,} | {sum(v['negative_cells'].values()):,} |")
L += ["", "## Label distribution (after label-string normalization)", ""]
L += [f"- {k}: {v:,}" for k, v in manifest["label_distribution"].items()]
L += ["", f"Cross-file duplicate feature vectors: {manifest['cross_file_duplicate_hashes']:,}",
      f"Feature vectors carrying conflicting labels: {manifest['label_conflict_hashes']:,} "
      f"({manifest['rows_in_label_conflict_hashes']:,} rows)"]
(REP / "data_quality_report.md").write_text("\n".join(L))
print("\n".join(L))
