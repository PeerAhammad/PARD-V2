"""PARD-SSM intrusion-detection research demo (Streamlit). Run from the project root:  streamlit run demo/app.py"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="PARD-SSM IDS demo", layout="wide")
DEMO, RES = Path("demo"), Path("results")
NONNEG = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s"]

@st.cache_data
def load_scores(): return pd.read_parquet(DEMO / "scores.parquet")
@st.cache_data
def load_meta(): return json.loads((DEMO / "meta.json").read_text())
@st.cache_resource
def load_model():
    import joblib
    return joblib.load(DEMO / "model.joblib")
def latest_final():
    d = sorted([p for p in RES.glob("final_*") if (p / "final_summary.csv").exists()])
    return d[-1] if d else None

def cm_figure(tn, fp, fn, tp, title=""):
    C = np.array([[tn, fp], [fn, tp]], float); N = C / np.maximum(C.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.6, 3.2)); ax.imshow(N, cmap="Blues", vmin=0, vmax=1)
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{N[r, c]:.1%}\n({int(C[r, c]):,})", ha="center", va="center",
                    color="white" if N[r, c] > .5 else "black", fontsize=9)
    ax.set_xticks([0, 1], ["Benign", "Attack"]); ax.set_yticks([0, 1], ["Benign", "Attack"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9); fig.tight_layout()
    return fig

st.sidebar.title("PARD-SSM IDS")
page = st.sidebar.radio("Page", ["Overview", "Live replay", "Results", "Score your own CSV"])
st.sidebar.info("Research demo on CICIDS2017. Not a production IDS.")
if not (DEMO / "scores.parquet").exists():
    st.error("Demo assets not found. Run:  python src/build_demo_assets.py"); st.stop()
meta, sc = load_meta(), load_scores()
thr = meta["threshold"]

# ------------------------------------------------------------------ Overview
if page == "Overview":
    st.title("Switching-state-space attack detection on CICIDS2017")
    st.write("A Random Forest flow classifier, evaluated under chronological and leave-family-out protocols, "
             "with causal temporal features and a 4-regime HMM (PARD-SSM) tested as add-ons.")
    e = meta["test_eval"]
    c = st.columns(4)
    c[0].metric("Recall (in-distribution test)", f"{e['recall']:.3f}")
    c[1].metric("False-positive rate", f"{e['fpr']:.4f}")
    c[2].metric("Accuracy", f"{e['accuracy']:.4f}")
    c[3].metric("Features", meta["n_features"])
    fin = latest_final()
    if fin is not None:
        s = pd.read_csv(fin / "final_summary.csv")
        l = s[(s.variant == "base") & (s.fold != "per_episode:test")]
        st.metric("Recall on UNSEEN attack families (macro-average, 14 held-out units)", f"{l.recall.mean():.3f}",
                  delta=f"{l.recall.mean() - e['recall']:+.3f} vs in-distribution", delta_color="off")
    st.subheader("Key findings")
    st.markdown("""
- In-distribution evaluation (chronological blocks within each attack episode) makes the Random Forest look near-perfect.
- When an entire attack family is held out, recall drops sharply, and the threshold calibrated on other days transfers badly
  (false-positive rate far above the 1% target on several folds).
- Causal temporal features did not help (worse on 12 of 14 folds); the 4-regime HMM (PARD-SSM) gave no operationally
  meaningful gain, so the simpler base model is the reported one.
- No early-detection claim is made: the data has no timestamps and no defined attack onset.
""")
    st.subheader("Protocol")
    st.markdown(f"""
- Data: 2,830,743 flows, {meta['n_features']} features; row order is the time proxy (no Timestamp column in this release).
- Splits: per-episode 60/20/20 blocks with embargo gaps; label-conflict vectors and exact train duplicates removed from evaluation.
- Operating point: threshold = 99th percentile of validation benign scores (1% FPR) = `{thr:.4f}`; chosen on validation only.
""")

# ------------------------------------------------------------------ Replay
elif page == "Live replay":
    st.title("Replay a held-out test block")
    st.caption("Flows are replayed in row order. Model: base RF, threshold fixed at 1% validation FPR. "
               "This is the in-distribution protocol; see Results for unseen-family performance.")
    t = sc[(sc.split == "test") & ~sc.conflict & ~sc.leak_in_train]
    eps = sorted(e for e in t.episode.unique() if (t[t.episode == e].Label != "BENIGN").any())
    c1, c2 = st.columns(2)
    sel = c1.selectbox("Episode (file | attack family)", eps)
    chunk = c2.slider("Flows per step", 200, 5000, 1000, 200)
    e = t[t.episode == sel].sort_values("row_in_file").reset_index(drop=True)
    n = int(np.ceil(len(e) / chunk))
    pos = st.slider("Position (step)", 1, n, n) if n > 1 else 1
    play = st.button("Play from start")
    ph = st.empty()

    def render(p):
        sub = e.iloc[:p * chunk]; ya = (sub.Label != "BENIGN").to_numpy(); al = sub.alert.to_numpy()
        tp, fn, fp, tn = int((al & ya).sum()), int((~al & ya).sum()), int((al & ~ya).sum()), int((~al & ~ya).sum())
        grp = (np.arange(len(sub)) // chunk)
        tl = pd.DataFrame({"attack fraction": ya, "alert rate": al}).groupby(grp).mean()
        with ph.container():
            m = st.columns(5)
            m[0].metric("Flows seen", f"{len(sub):,}"); m[1].metric("Attacks", f"{int(ya.sum()):,}")
            m[2].metric("Alerts", f"{int(al.sum()):,}")
            m[3].metric("Recall so far", f"{tp / max(tp + fn, 1):.3f}")
            m[4].metric("FPR so far", f"{fp / max(fp + tn, 1):.4f}")
            a, b = st.columns([2, 1])
            fig, ax = plt.subplots(figsize=(8, 3.2)); ax.fill_between(tl.index, tl["attack fraction"], color="lightgray", label="true attack fraction")
            ax.plot(tl.index, tl["alert rate"], label="alert rate"); ax.set_xlim(0, max(n, 1)); ax.set_ylim(0, 1.02)
            ax.set_xlabel("step"); ax.legend(fontsize=8); fig.tight_layout(); a.pyplot(fig); plt.close(fig)
            b.pyplot(cm_figure(tn, fp, fn, tp, "Cumulative confusion")); plt.close("all")
    if play:
        for p in range(1, n + 1):
            render(p); time.sleep(0.12)
    else:
        render(pos)
    fam = e[e.Label != "BENIGN"].groupby("Label").alert.agg(["mean", "size"]).rename(columns={"mean": "recall", "size": "attack flows"})
    if len(fam): st.subheader("Recall by family (whole block)"); st.dataframe(fam.style.format({"recall": "{:.3f}"}))

# ------------------------------------------------------------------ Results
elif page == "Results":
    st.title("Evaluation results")
    fin = latest_final()
    if fin is None:
        st.warning("No results\\final_* folder with final_summary.csv found."); st.stop()
    st.caption(f"Source: {fin}")
    s = pd.read_csv(fin / "final_summary.csv")
    cols = [c for c in ["roc_auc", "recall", "fpr", "precision", "f1", "accuracy", "recall_oracle"] if c in s]
    st.subheader("Per fold and variant (mean over seeds)")
    st.dataframe(s.set_index(["fold", "variant"])[cols].style.format("{:.4f}"))
    lfo = s[(s.variant != "") & (s.fold != "per_episode:test")].groupby("variant")[["recall", "fpr", "roc_auc", "recall_oracle"]].mean()
    st.subheader("Leave-family-out macro-average"); st.dataframe(lfo.style.format("{:.4f}"))
    figs = sorted((fin / "figures").glob("*.png")) if (fin / "figures").exists() else []
    if not figs: st.info("Run  python src/phase5_plots.py --run " + str(fin) + "  to generate figures.")
    for f in figs:
        st.subheader(f.stem.replace("_", " ")); st.image(str(f))

# ------------------------------------------------------------------ Score CSV
else:
    st.title("Score your own flows")
    st.caption("Upload a CICIDS2017-style CSV (same 78 feature columns; Label optional). Rows are scored with the base RF.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        df = pd.read_csv(up, encoding="latin-1", low_memory=False, nrows=300_000)
        df.columns = df.columns.str.strip()
        mdl = load_model(); feats = mdl["feats"]
        miss = [c for c in feats if c not in df.columns]
        if miss:
            st.error(f"Missing {len(miss)} feature columns, e.g. {miss[:5]}"); st.stop()
        X = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        for c in NONNEG:
            if c in X: X.loc[X[c] < 0, c] = np.nan
        s = mdl["clf"].predict_proba(mdl["imputer"].transform(X.to_numpy("float32")).astype("float32"))[:, 1]
        al = s > mdl["threshold"]
        st.write(f"Scored {len(df):,} flows; **{int(al.sum()):,} alerts** ({al.mean():.2%}) at threshold {mdl['threshold']:.4f}.")
        fig, ax = plt.subplots(figsize=(7, 2.8)); ax.hist(s, bins=60, log=True)
        ax.axvline(mdl["threshold"], c="k", ls="--"); ax.set_xlabel("attack score"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        if "Label" in df:
            ya = (df["Label"].astype(str).str.strip() != "BENIGN").to_numpy()
            tp, fn, fp, tn = int((al & ya).sum()), int((~al & ya).sum()), int((al & ~ya).sum()), int((~al & ~ya).sum())
            st.write(f"Recall {tp / max(tp + fn, 1):.3f}, FPR {fp / max(fp + tn, 1):.4f}")
            st.pyplot(cm_figure(tn, fp, fn, tp, "Uploaded file")); plt.close("all")
        out = df.assign(attack_score=s, alert=al).sort_values("attack_score", ascending=False)
        st.dataframe(out[["attack_score", "alert"] + ([ "Label"] if "Label" in df else [])].head(50))
        st.download_button("Download scored CSV", out.to_csv(index=False).encode(), "scored_flows.csv")