# Protocol v1 (frozen before any model run)
- Data: 69-feature CICIDS2017 (MachineLearningCVE release), original labels; time proxy = row order within file.
- Task: binary benign vs attack; per-family recall reported. Heartbleed, Infiltration, SQL Injection = rare, aggregate report only, never trained on.
- Splits: splits_v1 (per-episode chronological 60/20/20 by attack-row quantile, embargo <=1000 rows).
- Exclusions: label-conflict vectors never trained on or scored. Training rows deduplicated by feature hash.
- Primary eval set: "filtered" = val/test rows with no exact feature match in training. Sensitivity: "nonconflict" (no leak filter).
- Preprocessing (median imputation) fitted on training rows only.
- Operating point: threshold = 99th percentile of validation benign scores (1% FPR), chosen on validation only, then applied to test.
- Metrics: PR-AUC and recall@1%FPR primary; ROC-AUC, precision, F1, achieved FPR, per-family recall secondary. 5 seeds, mean and std.
- Test access: only via --eval-test, each use logged in results/test_access_log.csv. Final model comparison uses test once.
- SSM decision rule: kept only if it beats the same model without SSM features on the primary metrics on both the per-episode protocol and the leave-family-out protocol; otherwise the simpler model is reported.
- Early-warning claims: none unless an onset and alert protocol is defined and met (needs real timestamps).

## Addendum v1.2: Phase 3 temporal features (written before any Phase 3 run)
- Features (causal: each row sees itself and earlier rows of its own file in row order, never later rows):
  diff = first difference of 7 core features; roll = rolling mean/std (windows 50, 200, 1000) of the same 7;
  ind = rolling fraction (same windows) of: <=2-packet flows, SYN-without-ACK flows, flows with no backward packets;
  port = distinct destination ports in the last 200 and 1000 rows, divided by window size.
- Core features: Flow Duration, Total Fwd Packets, Total Backward Packets, Flow Bytes/s, Flow Packets/s, Flow IAT Mean, Packet Length Mean.
- Evaluation: 14 distinct LFO dev folds (11 families + tue, wed, thu_am_web) plus a per-episode validation control. Test blocks are not used in Phase 3.
- Comparison: RF base vs RF base+temporal. Same training rows, threshold rule, and seeds. Controls: same features computed on shuffled row order (null); per-episode protocol (no regression where baseline is at ceiling).
- Paired seed-0 block-bootstrap 95% CIs on delta recall (transferred threshold), delta FPR, delta ROC-AUC.
- Verdict per fold: "better" if delta-recall CI lower bound > 0 and delta-FPR CI upper bound <= +0.005; "worse" if delta-recall CI upper bound < 0 or delta-FPR CI lower bound > +0.005; otherwise "no clear change".
- Temporal features are credited only if they win on several folds AND the shuffled-order control does not show the same gains. Otherwise the simpler base model is retained.

## Addendum v1.3: Phase 4 minimal PARD-SSM (written before any Phase 4 run)
- Model: 4-regime Gaussian HMM (diagonal covariance), observations = 7 core flow features (signed log1p, standardised on fold-train rows), fitted by EM on fold-train rows only (random_state=0), per-file/episode sequences.
- Features added to the base RF: causal forward-filtered regime posteriors (4) + one-step predictive surprise (1). Filtered, never smoothed.
- Variants: base, ssm, ssm_shuffled (same filter run over shuffled row order: null control). Same folds, training rows, threshold rule and verdict rule as Phase 3.
- SSM is retained only if it is "better" on several LFO folds, ssm_shuffled is not similarly better, and the per-episode control does not regress. Otherwise the base RF is the reported model.

## Addendum v1.4: final evaluation (written before the test blocks were scored)
- Models: base RF (primary) and RF+SSM (secondary, reported for the ablation). Choice of primary fixed before testing, from the dev results.
- Test data: test block of the held-out unit in each of the 14 LFO folds, plus the per-episode test blocks. Threshold from validation benign rows only (1% FPR). Run once, with --confirm-final; each use is logged.
- Incident-level: attack rows <=500 rows apart form one incident; detected if >=3 flows alerted. No lead-time or early-warning claim (no timestamps, no defined onset).
- Reported: precision, recall, F1, ROC-AUC, PR-AUC, FPR, accuracy, confusion matrix, per-family recall, oracle recall, block-bootstrap 95% CIs, paired ssm-minus-base deltas, 3 seeds.