"""Phase 1: CICIDS2017 verification. Read-only on raw data. Writes reports/ only."""
import hashlib, json, sys, platform
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data")
OUT = Path("reports"); OUT.mkdir(exist_ok=True)

EXPECTED = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

def sha256(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(n):
            h.update(chunk)
    return h.hexdigest()

def load(p):
    df = pd.read_csv(p, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip()          # known: leading spaces in names
    return df

report = {"env": {"python": sys.version, "pandas": pd.__version__,
                  "numpy": np.__version__, "platform": platform.platform()},
          "files": {}}
frames, col_sets = [], {}

for name in EXPECTED:
    p = DATA / name
    if not p.exists():
        report["files"][name] = {"MISSING": True}; continue
    df = load(p)
    df["__src"] = name
    df["__row"] = np.arange(len(df))             # original row order in file
    col_sets[name] = tuple(c for c in df.columns if not c.startswith("__"))
    num = df.select_dtypes(include=[np.number])
    feat_cols = [c for c in df.columns if c not in ("Label", "__src", "__row")]
    # coerce non-numeric feature cols to see hidden junk
    coerced = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    report["files"][name] = {
        "sha256": sha256(p),
        "rows": int(len(df)),
        "n_columns": len(col_sets[name]),
        "labels": df["Label"].value_counts().to_dict(),
        "nan_cells": int(coerced.isna().sum().sum()),
        "inf_cells": int(np.isinf(coerced.to_numpy(dtype="float64", na_value=np.nan)).sum()),
        "negative_cells_in_duration_or_bytes": int(
            (coerced[[c for c in feat_cols if c in
              ("Flow Duration","Flow Bytes/s","Flow Packets/s")]] < 0).sum().sum()),
        "exact_duplicate_rows": int(df.drop(columns=["__src","__row"]).duplicated().sum()),
        "timestamp_sample": df["Timestamp"].head(3).tolist() if "Timestamp" in df else None,
    }
    frames.append(df)

# schema consistency across files
ref = next(iter(col_sets.values()))
report["schema_identical_across_files"] = all(c == ref for c in col_sets.values())
report["n_feature_columns_incl_label"] = len(ref)
report["duplicate_column_names"] = [c for c in ref if ref.count(c) > 1]

all_df = pd.concat(frames, ignore_index=True)
report["total_rows"] = int(len(all_df))
report["label_distribution_total"] = all_df["Label"].value_counts().to_dict()
report["label_by_file"] = (all_df.groupby(["__src", "Label"]).size()
                           .unstack(fill_value=0).to_dict(orient="index"))

# cross-file duplicates (same flow in two files)
key_cols = [c for c in ref if c != "Label"]
report["cross_file_duplicate_rows"] = int(
    all_df.duplicated(subset=key_cols, keep=False).sum()
    - all_df.duplicated(subset=key_cols + ["__src"], keep=False).sum())

# constant / near-constant features
feats = all_df[[c for c in key_cols if c != "Timestamp"]].apply(pd.to_numeric, errors="coerce")
report["constant_columns"] = [c for c in feats.columns if feats[c].nunique(dropna=True) <= 1]

# timestamp parsing check (known ambiguity: 12h format without AM/PM)
if "Timestamp" in all_df:
    ts = pd.to_datetime(all_df["Timestamp"], errors="coerce", dayfirst=False)
    report["timestamp_parse_failures"] = int(ts.isna().sum())
    mono = (all_df.assign(_t=ts).groupby("__src")["_t"]
            .apply(lambda s: float((s.diff().dropna() < pd.Timedelta(0)).mean())))
    report["fraction_nonmonotonic_timestamps_by_file"] = mono.to_dict()

(OUT / "data_quality_report.json").write_text(json.dumps(report, indent=2, default=str))
print(json.dumps({k: report[k] for k in
      ["total_rows","schema_identical_across_files","n_feature_columns_incl_label",
       "cross_file_duplicate_rows","constant_columns","label_distribution_total",
       "fraction_nonmonotonic_timestamps_by_file"] if k in report}, indent=2, default=str))