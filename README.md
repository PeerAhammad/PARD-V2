<div align="center">

PARD-V2

Evaluating Network Intrusion Detection Beyond Random Splits

A CICIDS2017 research pipeline for chronological evaluation, unseen attack-family testing, and a Random Forest vs. Switching State-Space Model comparison.

<p>
  <img src="https://img.shields.io/badge/Domain-Cybersecurity-1f6feb?style=for-the-badge" alt="Domain: Cybersecurity">
  <img src="https://img.shields.io/badge/Task-Intrusion%20Detection-8250df?style=for-the-badge" alt="Task: Intrusion Detection">
  <img src="https://img.shields.io/badge/Dataset-CICIDS2017-238636?style=for-the-badge" alt="Dataset: CICIDS2017">
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-d29922?style=for-the-badge" alt="Status: Research Prototype">
</p>

Repository · Python · scikit-learn · pandas · Streamlit

</div>

Overview

High test scores can hide a serious weakness in intrusion detection: a model may perform well on data similar to its training set while missing attack families it has never seen.

PARD-V2 studies that gap. It builds a reproducible evaluation pipeline on CICIDS2017 and compares a Random Forest (RF) baseline with a Switching State-Space Model (SSM) add-on. The experiments include chronological evaluation and leave-family-out (LFO) testing, with metrics reported beyond accuracy.

Main finding: Strong in-distribution performance did not translate into reliable detection of every held-out attack family, and the SSM add-on did not consistently outperform the simpler Random Forest baseline.

This is an evaluation-focused research project—not a claim that one model solves general-purpose intrusion detection.

At a glance

Project detail

Description

Dataset

CICIDS2017 network-flow records

Verified rows

2,830,743

Demo model inputs

69 features

Baseline

Random Forest

Comparison

Random Forest with a Switching State-Space Model add-on

Evaluation

Chronological splits and leave-family-out tests

Included outputs

Metrics, family-level results, plots, protocol notes, and a Streamlit demo

Results

1. In-distribution test

The table below reports the final per_episode:test results. These figures describe the specified test protocol; they are not estimates of performance on unseen attack families.

Metric

Random Forest

RF + SSM

ROC-AUC

0.9970

0.9970

PR-AUC

0.9955

0.9955

Recall

0.9950

0.9950

Precision

0.9311

0.9268

F1-score

0.9620

0.9597

False-positive rate

0.0103

0.0110

Accuracy

0.9903

0.9897

Interpretation: On this evaluation, the SSM add-on did not improve the headline discrimination metrics and slightly reduced precision, F1-score, and accuracy while increasing the false-positive rate. The results do not support claiming a consistent benefit from adding the SSM.

2. Generalization to held-out attack families

Leave-family-out testing asks a different question from a conventional test split: what happens when an entire attack family is excluded from training?

The final results show that performance varies substantially by family. In particular:

Bot: the baseline recorded zero recall on the held-out family.

PortScan: baseline recall was very low (approximately 0.021).

SSM contribution: improvements were mixed across evaluations rather than consistent.

These results are important even when the in-distribution scores are high. Aggregate metrics can obscure failure on specific attack families.

Result figures

The final figures are included in results/final_20261028_172705_final/figures/.

Figure

What it shows

Per-episode confusion matrix

Confusion counts for the per-episode evaluation

Leave-family-out confusion matrix

Baseline errors under held-out-family evaluation

Recall by attack family

Differences in recall across families

LFO recall and false-positive rate

Family-level detection and false-positive behaviour

SSM minus baseline

Metric differences between the SSM variant and baseline

Incident detection

Incident-level evaluation output

For machine-readable results, start with final_summary.csv, final_family_recall.csv, and final_metrics_per_seed.csv.

Research questions

PARD-V2 is organized around four questions:

How does a supervised flow classifier perform under chronological evaluation?

How much does performance change when a complete attack family is held out?

Does adding a Switching State-Space Model improve detection over the Random Forest baseline?

What do precision, recall, false-positive rate, family-level results, and incident-level evaluation reveal that accuracy alone can miss?

Methodology

The repository separates the experimental work into phases:

Data verification — inspect dataset consistency and basic quality issues.

Split construction — create evaluation splits and document the protocol.

Baseline evaluation — train and evaluate Random Forest models.

Leave-family-out evaluation — assess performance on held-out attack families.

Temporal experiments — evaluate additional temporal features and settings.

SSM experiments — test whether the switching-state-space component adds value.

Final evaluation — export metrics, family-level summaries, and figures.

Important evaluation note

Where verified event timestamps are unavailable, row order is used only as a time proxy. Row order is not equivalent to verified event time, and conclusions from these splits should be interpreted accordingly. Review the protocol and configuration files before reproducing or comparing individual results.

Interactive demo

The included Streamlit app provides a way to inspect the model and evaluation outputs.

Demo capabilities

Model overview and reported metrics

Replay of held-out test-flow scores

Results tables and plots

Scoring for compatible CICIDS2017-style CSV files

Download of scored CSV output

Run locally

Run these commands from the repository root in PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn hmmlearn joblib streamlit matplotlib pyarrow
streamlit run demo/app.py

The app expects the supporting assets in demo/ and final evaluation outputs in results/final_20261028_172705_final/. For CSV uploads, use compatible feature columns expected by the model. The app limits uploads to 300,000 rows.

The demo is a research interface for exploring the included artifacts. It is not a live network sensor or a production deployment.

Repository structure

PARD-V2/
├── demo/
│   ├── app.py
│   ├── meta.json
│   ├── model.joblib
│   └── scores.parquet
├── reports/
│   ├── protocol_v1.md
│   └── split_summary_v1.md
├── results/
│   └── final_20261028_172705_final/
│       ├── final_summary.csv
│       ├── final_metrics_per_seed.csv
│       ├── final_family_recall.csv
│       └── figures/
├── src/
│   ├── phase1_checks.py
│   ├── phase1_splits.py
│   ├── phase2_baselines.py
│   ├── phase2_plots.py
│   ├── phase2b_leave_family_out.py
│   ├── phase3_temporal.py
│   ├── phase4_ssm.py
│   ├── phase5_final_test.py
│   ├── phase5_plots.py
│   ├── build_demo_assets.py
│   └── make_test_csv.py
└── phase1_verify.py

Reproducibility

The repository includes the experiment scripts, protocol notes, final metric tables, plots, and environment metadata. For a reproducible comparison:

Obtain CICIDS2017 from its authoritative distribution.

Review reports/protocol_v1.md and reports/split_summary_v1.md.

Inspect the configuration and environment metadata in the final results directory.

Run the relevant phase scripts and compare outputs against the committed result tables.

The raw dataset is not included in this repository. Dataset access and use remain subject to its applicable terms.

Limitations

High in-distribution performance does not guarantee detection of unseen attack families.

Some held-out families have very low or zero recall.

The SSM add-on does not consistently outperform the baseline in the reported experiments.

Row order is only a proxy for time where verified timestamps are unavailable.

Results depend on the dataset, split protocol, feature processing, thresholds, and evaluation setup.

The included demo and model artifacts are for research and should not be treated as production-ready security controls.

Responsible use

PARD-V2 is intended for academic research, evaluation, and defensive cybersecurity experimentation. Do not use the reported metrics as a guarantee of real-world detection performance. Validate any model independently in the environment where it would be used.

<div align="center">

PARD-V2 · Researching the gap between strong test scores and robust generalization

</div>