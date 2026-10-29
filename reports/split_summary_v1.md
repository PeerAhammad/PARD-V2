# Split manifest v1

## All rows by split

split                       embargo  rare_report    test    train     val
Label
BENIGN                        16817            0  602930  1317674  335676
Bot                             193            0     385     1179     209
DDoS                           2000            0   24606    76816   24605
DoS GoldenEye                    92            0    2013     6175    2013
DoS Hulk                       1936            0   45255   138643   45239
DoS Slowhttptest                221            0     990     3299     989
DoS slowloris                   625            0    1075     3477     619
FTP-Patator                     321            0    1390     4762    1465
Heartbleed                        0           11       0        0       0
Infiltration                      0           36       0        0       0
PortScan                       1855            0   30919    95358   30798
SSH-Patator                      38            0    1161     3538    1160
Web Attack - Brute Force         77            0     271      904     255
Web Attack - Sql Injection        0           21       0        0       0
Web Attack - XSS                 47            0     119      391      95

## Eligible val/test rows (no label conflict, no exact-match with train)

split                       test     val
Label
BENIGN                    557909  300782
Bot                          385     209
DDoS                       24601   24604
DoS GoldenEye               2013    2013
DoS Hulk                   32690   32901
DoS Slowhttptest             990     989
DoS slowloris               1041     538
FTP-Patator                  946    1049
PortScan                   14511   16567
SSH-Patator                  634     605
Web Attack - Brute Force     263     245
Web Attack - XSS             119      95

Rare families (report-only): ['Heartbleed', 'Infiltration', 'Web Attack - Sql Injection']
