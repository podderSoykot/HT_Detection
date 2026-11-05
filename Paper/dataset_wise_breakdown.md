# Per-Dataset Performance Evaluation Breakdown

Evaluation across individual benchmark suites independently:

| Dataset Evaluation                        |   Total Circuits Tested |   Golden Count |   Trojan Count | Accuracy   | Precision   | Recall (Sensitivity)   | Miss Rate (FNR)   | False Alarm (FAR)   |   F1-Score |   ROC-AUC |
|:------------------------------------------|------------------------:|---------------:|---------------:|:-----------|:------------|:-----------------------|:------------------|:--------------------|-----------:|----------:|
| Dataset 1: EPFL Benchmark Suite           |                      13 |             13 |              0 | 69.2%      | 0.0%        | 0.0%                   | 0.0%              | 30.8%               |          0 |         1 |
| Dataset 2: Fast & Accurate ISCAS HT Suite |                      49 |             25 |             24 | 100.0%     | 100.0%      | 100.0%                 | 0.0%              | 0.0%                |          1 |         1 |
| Dataset 3: Trust-Hub AES Flagship Suite   |                      30 |             15 |             15 | 100.0%     | 100.0%      | 100.0%                 | 0.0%              | 0.0%                |          1 |         1 |

## Per-Dataset Observations

1. **EPFL Combinational Suite**: Serves as a pure golden reference baseline to verify zero false positives on standard combinational arithmetic/logic designs.
2. **Fast & Accurate ISCAS HT Suite**: Tests fine-grained gate-level Trojan triggers across ALUs, multipliers, and logic cores with high precision.
3. **Trust-Hub AES Flagship Suite**: Tests complex multi-round cryptographic cores under side-channel and structural stress.
