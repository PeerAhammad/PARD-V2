"""Figures + tables from saved baseline scores. No model fitting; reads scores.npz only."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--split", default="test")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
R = Path(a.run); F = R / "figures"; F.mkdir(exist_ok=True)
Z = np.load(R / "scores.npz"); THR = json.load(open(R / "thresholds.json"))
y, fam, spl, leak, tag, row = (Z[k] for k in ["y", "fam", "split", "leak", "file_tag", "row"])
m = (spl == a.split) & ~leak                      # filtered set, as in the protocol
MODELS = {"rf": "Random Forest", "iforest": "Isolation Forest"}
S = {k: Z[f"{k}_{a.seed}"] for k in MODELS}; T = {k: THR[f"{k}_{a.seed}"] for k in MODELS}

def wilson(k, n, z=1.96):
    if n == 0: return np.nan, np.nan
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

# 1) confusion matrices
summary = []
for k, name in MODELS.items():
    pred = S[k][m] > T[k]; yy = y[m]
    C = np.array([[((yy == 0) & ~pred).sum(), ((yy == 0) & pred).sum()],
                  [((yy == 1) & ~pred).sum(), ((yy == 1) & pred).sum()]])
    acc = (C[0, 0] + C[1, 1]) / C.sum()
    summary.append(dict(model=name, accuracy=acc, tn=C[0, 0], fp=C[0, 1], fn=C[1, 0], tp=C[1, 1]))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, (M, title, fmt) in enumerate([(C, "Counts", "{:,.0f}"), (C / C.sum(1, keepdims=True), "Row-normalised", "{:.4f}")]):
        ax[i].imshow(M, cmap="Blues")
        for r in range(2):
            for c in range(2):
                ax[i].text(c, r, fmt.format(M[r, c]), ha="center", va="center",
                           color="white" if M[r, c] > M.max() / 2 else "black")
        ax[i].set_xticks([0, 1], ["Benign", "Attack"]); ax[i].set_yticks([0, 1], ["Benign", "Attack"])
        ax[i].set_xlabel("Predicted"); ax[i].set_ylabel("True"); ax[i].set_title(title)
    fig.suptitle(f"{name}: {a.split} (filtered), threshold at 1% val FPR, accuracy {acc:.4f}")
    fig.tight_layout(); fig.savefig(F / f"confusion_{k}_{a.split}.png", dpi=150); plt.close(fig)
pd.DataFrame(summary).to_csv(F / "confusion_summary.csv", index=False)
print(pd.DataFrame(summary).round(4).to_string(index=False))

# 2) ROC (log FPR axis) and PR curves
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for k, name in MODELS.items():
    s = S[k][m]; fpr, tpr, _ = roc_curve(y[m], s); pr, rc, _ = precision_recall_curve(y[m], s)
    ax[0].plot(fpr, tpr, label=name); ax[1].plot(rc, pr, label=name)
    pred = s > T[k]; ax[0].scatter((pred & (y[m] == 0)).sum() / (y[m] == 0).sum(), pred[y[m] == 1].mean(), zorder=5)
ax[0].set_xscale("log"); ax[0].set_xlim(1e-4, 1); ax[0].set_xlabel("False-positive rate (log)"); ax[0].set_ylabel("Recall")
ax[0].set_title("ROC (dots = operating point)"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("PR")
ax[1].axhline(y[m].mean(), ls=":", c="gray", label="Majority/prevalence")
for x in ax: x.legend(); x.grid(alpha=.3)
fig.tight_layout(); fig.savefig(F / f"roc_pr_{a.split}.png", dpi=150); plt.close(fig)

# 3) per-family recall with Wilson 95% CI (+ table)
fams = sorted(set(fam[m & (y == 1)])); rows = []
for f in fams:
    mm = m & (fam == f)
    for k, name in MODELS.items():
        hit = int((S[k][mm] > T[k]).sum()); n = int(mm.sum()); lo, hi = wilson(hit, n)
        rows.append(dict(family=f, model=name, n=n, detected=hit, missed=n - hit, recall=hit / n, ci_lo=lo, ci_hi=hi))
FR = pd.DataFrame(rows); FR.to_csv(F / f"family_recall_{a.split}.csv", index=False)
fig, ax = plt.subplots(figsize=(11, 4.8)); w = 0.4
for i, (k, name) in enumerate(MODELS.items()):
    d = FR[FR.model == name]
    ax.bar(np.arange(len(fams)) + (i - .5) * w, d.recall, w, label=name,
            yerr=[np.clip(d.recall - d.ci_lo, 0, None), np.clip(d.ci_hi - d.recall, 0, None)], capsize=2)
ax.set_xticks(range(len(fams)), [f"{f}\n(n={FR[FR.family == f].n.iloc[0]})" for f in fams], rotation=40, ha="right", fontsize=8)
ax.set_ylabel("Recall at 1% val-FPR"); ax.set_title(f"Per-family recall, {a.split} (95% Wilson CI)"); ax.legend(); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(F / f"family_recall_{a.split}.png", dpi=150); plt.close(fig)

# 4) score distributions: benign vs key families, threshold marked
groups = {"Benign": m & (y == 0), "DDoS": m & (fam == "DDoS"), "PortScan": m & (fam == "PortScan"),
          "DoS Hulk": m & (fam == "DoS Hulk"), "Bot": m & (fam == "Bot")}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for i, (k, name) in enumerate(MODELS.items()):
    lo, hi = np.percentile(S[k][m], [0.1, 99.9]); bins = np.linspace(lo, hi, 60)
    for g, mm in groups.items():
        if mm.sum(): ax[i].hist(np.clip(S[k][mm], lo, hi), bins=bins, histtype="step", density=True, label=f"{g} (n={mm.sum():,})")
    ax[i].axvline(T[k], c="k", ls="--", label="threshold"); ax[i].set_yscale("log")
    ax[i].set_title(f"{name} score, {a.split}"); ax[i].set_xlabel("score"); ax[i].legend(fontsize=7)
fig.tight_layout(); fig.savefig(F / f"score_dist_{a.split}.png", dpi=150); plt.close(fig)

# 5) detection over row order for the big episodes (val and test blocks only)
B = 2000
for t in ["fri_pm_ddos", "fri_pm_scan", "wed", "fri_am"]:
    mm = (tag == t) & np.isin(spl, ["val", "test"]) & ~leak
    if mm.sum() == 0: continue
    df = pd.DataFrame({"bin": row[mm] // B, "attack": y[mm], "rf": S["rf"][mm] > T["rf"], "iforest": S["iforest"][mm] > T["iforest"]})
    g = df.groupby("bin").mean(); g = g.reindex(range(g.index.min(), g.index.max() + 1))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.fill_between(g.index, g.attack, color="lightgray", label="true attack fraction")
    ax.plot(g.index, g.rf, label="RF alert rate"); ax.plot(g.index, g.iforest, label="IF alert rate")
    for sp, col in [("val", "tab:green"), ("test", "tab:red")]:
        r = row[(tag == t) & (spl == sp)]
        if len(r): ax.axvspan(r.min() // B, r.max() // B, color=col, alpha=.06, label=f"{sp} block")
    ax.set_xlabel(f"row index / {B}"); ax.set_ylabel("rate per bin"); ax.set_title(f"{t}: detection over row order"); ax.legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(F / f"timeline_{t}.png", dpi=150); plt.close(fig)
print("figures in", F)