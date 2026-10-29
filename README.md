# PARD-V2: Network Intrusion Detection on CICIDS2017

**PARD-V2** is a research project investigating network intrusion detection under chronological evaluation and leave-family-out testing. It compares a Random Forest baseline with a switching-state-space-model (SSM) add-on.

The project emphasizes generalization to unseen attack families rather than relying exclusively on conventional random train-test splits.

## Research Questions

* How well does a supervised flow classifier generalize across chronological data splits?
* How does detection performance change when an entire attack family is held out?
* Does adding switching-state-space modeling improve detection performance?
* What limitations are revealed by recall, precision, false-positive rate, and incident-level evaluation?

## Dataset

* **Dataset:** CICIDS2017
* **Verified flow rows:** 2,830,743
* **Features:** 69 model features in the demo
* **Evaluation:** Chronological and leave-family-out protocols

The project uses CICIDS2017 flow records. Row order is used as a time proxy where applicable; this is not equivalent to having verified event timestamps.

The original dataset is not included in this repository. Obtain it from its authoritative distribution and follow the dataset's applicable terms.

## Methodology

The experimental pipeline includes:

1. Data verification and preprocessing checks.
2. Chronological split construction and leakage checks.
3. Random Forest baseline evaluation.
4. Leave-family-out evaluation.
5. Temporal-feature experiments.
6. Switching-state-space-model experiments.
7. Final evaluation, metrics, and visualization.

The scripts and reports document the implemented protocols. Consult them before interpreting or reproducing individual experiments.

## Key Results

### In-distribution test

The reported baseline results for `per_episode:test` are:

| Metric              | Random Forest baseline |
| ------------------- | ---------------------: |
| ROC-AUC             |                 0.9970 |
| Recall              |                 0.9950 |
| Precision           |                 0.9311 |
| F1-score            |                 0.9620 |
| False-positive rate |                 0.0103 |
| Accuracy            |                 0.9903 |

These results apply to the specified in-distribution evaluation and should not be interpreted as performance on unseen attack families.

### Generalization and SSM findings

The leave-family-out results show substantial variation across attack families. In particular, the baseline has zero recall on the held-out Bot family and very low recall on PortScan.

The SSM add-on produces mixed results rather than a consistent improvement. For example, it improves some metrics on the Tuesday chronological fold but reduces recall on the Wednesday fold.

**Main conclusion:** Strong in-distribution performance does not establish reliable generalization to unseen attack families. The simpler baseline remains an important reference point, and the SSM component requires further evidence before it can be considered an operational improvement.

See `results/final_20261028_172705_final/` for the final metrics, CSV reports, and figures.

## Demo

A Streamlit research demo is provided in `demo/app.py`.

Features include:

* Model overview and reported metrics.
* Replay of held-out test flows.
* Evaluation tables and plots.
* Scoring of compatible CICIDS2017-style CSV files.
* Downloading scored CSV output.

Run the demo from the repository root after installing the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn hmmlearn joblib streamlit matplotlib pyarrow
streamlit run demo/app.py
```

The demo expects its supporting files in `demo/` and final evaluation outputs in `results/final_20261028_172705_final/`.

Use a compatible CSV containing the required model feature columns when testing the upload interface. The app limits uploads to 300,000 rows.

## Repository Structure

```text
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
```

## Limitations

* In-distribution performance does not guarantee generalization to unseen attacks.
* Some held-out attack families have very low or zero recall.
* The SSM component does not consistently outperform the baseline.
* Row order is only a proxy for time where timestamps are unavailable.
* This is a research prototype, not a production-ready intrusion detection system.
* Results depend on the dataset, split protocol, feature processing, threshold, and evaluation setup.

## Reproducibility

The repository includes experiment scripts, protocol documentation, final metrics, plots, and environment information. Review the documented configuration and split protocol before reproducing results.

## Disclaimer

This project is intended for academic research and defensive cybersecurity experimentation. It is not a guarantee of real-world detection performance.
