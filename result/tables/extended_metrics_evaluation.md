# Comprehensive 18-Metric Academic Evaluation Benchmark

| Metric Category & Name        |   Baseline 1: Graph GNN |   Baseline 2: Side-Channel |   Proposed: Multi-Modal Fusion |
|:------------------------------|------------------------:|---------------------------:|-------------------------------:|
| Accuracy (%)                  |                 87.5    |                    96.25   |                        98.75   |
| Balanced Accuracy (%)         |                 87.5    |                    96.25   |                        98.75   |
| Precision (PPV) (%)           |                 81.25   |                    95.12   |                        97.56   |
| Recall / Sensitivity (%)      |                 97.5    |                    97.5    |                       100      |
| Specificity (TNR) (%)         |                 77.5    |                    95      |                        97.5    |
| F1-Score                      |                  0.8864 |                     0.963  |                         0.9877 |
| Matthews Corr (MCC)           |                  0.7655 |                     0.9253 |                         0.9753 |
| Cohen's Kappa                 |                  0.75   |                     0.925  |                         0.975  |
| Youden's J-Index              |                  0.75   |                     0.925  |                         0.975  |
| False Alarm Rate (FAR) (%)    |                 22.5    |                     5      |                         2.5    |
| Miss Rate (FNR) (%)           |                  2.5    |                     2.5    |                         0      |
| False Discovery Rate (%)      |                 18.75   |                     4.88   |                         2.44   |
| Negative Pred Value (NPV) (%) |                 96.87   |                    97.44   |                       100      |
| ROC-AUC                       |                  0.8844 |                     0.9956 |                         1      |
| PR-AUC (Avg Precision)        |                  0.8944 |                     0.9956 |                         1      |
| Brier Calibration Loss        |                  0.1236 |                     0.0304 |                         0.0161 |
| Cross-Entropy Log-Loss        |                  0      |                     0      |                         0      |
| Inference Latency (ms)        |                  5.856  |                     0.001  |                         5.871  |
| Model Parameters              |              24676      |                    34      |                    135235      |

## Metric Definitions & Security Significance

- **Miss Rate (False Negative Rate, FNR)**: In Hardware Trojan detection, Miss Rate is safety-critical. A missed Trojan represents an undetected malicious payload embedded in silicon.
- **False Alarm Rate (FAR / FPR)**: Measures unnecessary quarantine or scrap of expensive, clean silicon wafers.
- **Matthews Correlation Coefficient (MCC)**: Balanced measure $(-1.0 \text{ to } +1.0)$ robust against extreme class imbalances.
- **Brier Calibration Loss**: Measures confidence calibration of output security probabilities ($0.0$ is perfect calibration).
- **Inference Latency (ms)**: Real-time silicon inspection speed per circuit netlist.
