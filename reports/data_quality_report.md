# CICIDS2017 data-quality report (Phase 1)

Total rows: 2,830,743. Features kept: 69. Time proxy: row order within file (no Timestamp column).

| file | rows | in-file dups | rows w/ invalid | inf cells | neg cells |
|---|---|---|---|---|---|
| mon | 529,918 | 26,935 | 452 | 810 | 39 |
| tue | 445,909 | 24,065 | 281 | 327 | 34 |
| wed | 692,703 | 81,957 | 1,318 | 1,586 | 58 |
| thu_am_web | 170,366 | 6,066 | 146 | 250 | 32 |
| thu_pm_inf | 288,602 | 35,630 | 211 | 396 | 12 |
| fri_am | 191,033 | 6,888 | 131 | 216 | 26 |
| fri_pm_scan | 286,467 | 72,353 | 407 | 727 | 108 |
| fri_pm_ddos | 225,745 | 2,633 | 36 | 64 | 6 |

## Label distribution (after label-string normalization)

- BENIGN: 2,273,097
- DoS Hulk: 231,073
- PortScan: 158,930
- DDoS: 128,027
- DoS GoldenEye: 10,293
- FTP-Patator: 7,938
- SSH-Patator: 5,897
- DoS slowloris: 5,796
- DoS Slowhttptest: 5,499
- Bot: 1,966
- Web Attack - Brute Force: 1,507
- Web Attack - XSS: 652
- Infiltration: 36
- Web Attack - Sql Injection: 21
- Heartbleed: 11

Cross-file duplicate feature vectors: 34,208
Feature vectors carrying conflicting labels: 698 (7,020 rows)