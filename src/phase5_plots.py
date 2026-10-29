"""Final-phase figures from the CSVs saved by phase5_final_test.py. No model fitting, no test re-scoring."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="results\\final_<timestamp>_<tag> folder")
a = ap.parse_args()
R = Path(a.run); F = R / "figures"; F.mkdir(exist_ok=True)
M = pd.read_csv(R / "final_metrics_per_seed.csv")
FR = pd.read_csv(R / "final_family_recall.csv")
CI = pd.read_csv(R / "final_block_bootstrap_ci_seed0.csv")
M0 = M[M.seed == 0]
NAMES = {"base": "Base RF", "ssm": "RF + SSM"}
short = lambda s: s.replace("family:", "").replace("Web Attack - ", "Web ").replace("DoS ", "").replace("per_episode:test", "PER-EPISODE")
lfo = [f for f in M.fold.unique() if f != "per_episode:test"]

def draw_cm(ax, tn, fp, fn, tp, title):
    C = np.array([[tn, fp], [fn, tp]], float); N = C / np.maximum(C.sum(1, keepdims=True), 1)
    ax.imshow(N, cmap="Blues", vmin=0, vmax=1)
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{N[r, c]:.1%}\n({int(C[r, c]):,})", ha="center", va="center",
                    color="white" if N[r, c] > .5 else "black", fontsize=8)
    ax.set_xticks([0, 1], ["Benign", "Attack"], fontsize=8); ax.set_yticks([0, 1], ["Benign", "Attack"], fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8); ax.set_title(title, fontsize=9)

# 1) per-episode confusion matrices (base and SSM), counts + row-normalised %
pe = M0[M0.fold == "per_episode:test"]
if len(pe):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for i, v in enumerate(["base", "ssm"]):
        r = pe[pe.variant == v].iloc[0]
        draw_cm(ax[i], r.tn, r.fp, r.fn, r.tp, f"{NAMES[v]}: accuracy {r.accuracy:.4f}")
    fig.suptitle("Per-episode TEST set (filtered), threshold at 1% validation FPR (cell: row %, count)", fontsize=10)
    fig.tight_layout(); fig.savefig(F / "confusion_per_episode.png", dpi=150); plt.close(fig)

# 2) leave-family-out confusion matrices, base RF, one panel per fold
n = len(lfo); cols = 4; rws = int(np.ceil(n / cols))
fig, ax = plt.subplots(rws, cols, figsize=(3.3 * cols, 3.2 * rws)); ax = np.atleast_1d(ax).ravel()
for i, f in enumerate(lfo):
    r = M0[(M0.fold == f) & (M0.variant == "base")].iloc[0]
    draw_cm(ax[i], r.tn, r.fp, r.fn, r.tp, f"{short(f)}  (recall {r.recall:.2f}, FPR {r.fpr:.3f})")
for j in range(n, len(ax)): ax[j].axis("off")
fig.suptitle("Leave-family-out TEST: base RF confusion matrices (held-out unit)", fontsize=11)
fig.tight_layout(); fig.savefig(F / "confusion_lfo_base.png", dpi=130); plt.close(fig)

# 3) recall and FPR by fold, base vs SSM (seed 0 with block-bootstrap 95% CI)
d = M0[M0.fold.isin(lfo)]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6)); x = np.arange(len(lfo)); w = 0.4
for i, v in enumerate(["base", "ssm"]):
    dv = d[d.variant == v].set_index("fold").loc[lfo]; cv = CI[CI.variant == v].set_index("fold").loc[lfo]
    ax[0].bar(x + (i - .5) * w, dv.recall, w, label=NAMES[v],
              yerr=[(dv.recall - cv.recall_lo).clip(lower=0), (cv.recall_hi - dv.recall).clip(lower=0)], capsize=2)
    ax[1].bar(x + (i - .5) * w, dv.fpr, w, label=NAMES[v],
              yerr=[(dv.fpr - cv.fpr_lo).clip(lower=0), (cv.fpr_hi - dv.fpr).clip(lower=0)], capsize=2)
ax[0].set_ylabel("Recall at transferred threshold"); ax[0].set_title("Leave-family-out recall (TEST)")
ax[1].set_ylabel("FPR on held-out benign"); ax[1].axhline(0.01, c="k", ls="--", lw=.8, label="1% target"); ax[1].set_title("Achieved FPR (TEST)")
for k in ax:
    k.set_xticks(x, [short(f) for f in lfo], rotation=50, ha="right", fontsize=8); k.legend(fontsize=8); k.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(F / "lfo_recall_fpr.png", dpi=150); plt.close(fig)

# 4) per-family recall: held-out (leave-family-out) vs in-distribution (per-episode)
fm = FR.groupby(["fold", "variant", "family"]).agg(recall=("recall", "mean"), n=("n", "first")).reset_index()
held = fm[fm.fold.str.startswith("family:") & (fm.fold.str.replace("family:", "") == fm.family)]
inep = fm[fm.fold == "per_episode:test"]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
for k, (dfm, ttl) in enumerate([(inep, "In-distribution (per-episode split)"), (held, "Held-out family (leave-family-out)")]):
    fams = sorted(dfm.family.unique()); xx = np.arange(len(fams))
    for i, v in enumerate(["base", "ssm"]):
        dv = dfm[dfm.variant == v].set_index("family").reindex(fams)
        ax[k].bar(xx + (i - .5) * w, dv.recall, w, label=NAMES[v])
    ns = dfm[dfm.variant == "base"].set_index("family").reindex(fams).n
    ax[k].set_xticks(xx, [f"{short(f)}\n(n={int(n_):,})" for f, n_ in zip(fams, ns)], rotation=50, ha="right", fontsize=7)
    ax[k].set_title(ttl); ax[k].grid(axis="y", alpha=.3); ax[k].legend(fontsize=8)
ax[0].set_ylabel("Recall (mean over seeds)")
fig.tight_layout(); fig.savefig(F / "family_recall_test.png", dpi=150); plt.close(fig)

# 5) incident-level detection (base RF, seed 0)
inc = M0[(M0.variant == "base")].set_index("fold").loc[lfo + (["per_episode:test"] if len(pe) else [])]
fig, ax = plt.subplots(figsize=(11, 4)); xx = np.arange(len(inc))
ax.bar(xx, inc.incidents, color="lightgray", label="incidents"); ax.bar(xx, inc.incidents_detected, label="detected")
ax.set_xticks(xx, [short(f) for f in inc.index], rotation=50, ha="right", fontsize=8)
ax.set_ylabel("Count"); ax.set_title("Incident-level detection, base RF (incident = attack rows <=500 rows apart; detected if >=3 alerts)")
ax.legend(); ax.grid(axis="y", alpha=.3); fig.tight_layout(); fig.savefig(F / "incident_detection.png", dpi=150); plt.close(fig)

# 6) SSM minus base, paired block-bootstrap 95% CIs
dd = CI[(CI.variant == "ssm")].set_index("fold").loc[lfo]
m0 = M0.set_index(["fold", "variant"])
dr = np.array([m0.loc[(f, "ssm"), "recall"] - m0.loc[(f, "base"), "recall"] for f in lfo])
da = np.array([m0.loc[(f, "ssm"), "roc_auc"] - m0.loc[(f, "base"), "roc_auc"] for f in lfo])
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4)); x = np.arange(len(lfo))
for k, (pt, lo, hi, ttl) in enumerate([(dr, dd.d_recall_lo, dd.d_recall_hi, "Delta recall (SSM - base)"),
                                       (da, dd.d_auc_lo, dd.d_auc_hi, "Delta ROC-AUC (SSM - base)")]):
    ax[k].bar(x, pt, yerr=[np.clip(pt - lo.to_numpy(), 0, None), np.clip(hi.to_numpy() - pt, 0, None)], capsize=2, color="tab:purple")
    ax[k].axhline(0, c="gray", lw=.8); ax[k].set_title(ttl + ", seed 0, 95% block bootstrap")
    ax[k].set_xticks(x, [short(f) for f in lfo], rotation=50, ha="right", fontsize=8); ax[k].grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(F / "ssm_minus_base.png", dpi=150); plt.close(fig)

# tables
M.groupby(["fold", "variant"])[["tp", "fn", "fp", "tn"]].mean().round(1).to_csv(F / "confusion_counts_mean_over_seeds.csv")
print("figures in", F)
for p in sorted(F.iterdir()): print(" ", p.name)