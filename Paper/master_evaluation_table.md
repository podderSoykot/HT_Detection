# Master Performance Evaluation Tables

Completely reconstructed evaluation tables reporting all 18+ academic, security error, calibration, and efficiency metrics across frameworks and individual datasets.

## Table 1: Comprehensive Multi-Modal Framework Ablation

| Performance Evaluation Metric   |   Baseline 1: Circuit Graph (GNN) |   Baseline 2: Side-Channel (Power/EM) |   Proposed: Multi-Modal Gated Fusion |
|:--------------------------------|----------------------------------:|--------------------------------------:|-------------------------------------:|
| Accuracy (%)                    |                           87.5    |                               98.75   |                              98.75   |
| Balanced Accuracy (%)           |                           87.5    |                               98.75   |                              98.75   |
| Precision (PPV) (%)             |                           81.25   |                               97.56   |                              97.56   |
| Recall / Sensitivity (%)        |                           97.5    |                              100      |                             100      |
| Specificity (TNR) (%)           |                           77.5    |                               97.5    |                              97.5    |
| F1-Score                        |                            0.8864 |                                0.9877 |                               0.9877 |
| Matthews Corr (MCC)             |                            0.7655 |                                0.9753 |                               0.9753 |
| Cohen's Kappa                   |                            0.75   |                                0.975  |                               0.975  |
| Youden's J-Index                |                            0.75   |                                0.975  |                               0.975  |
| False Alarm Rate (FAR) (%)      |                           22.5    |                                2.5    |                               2.5    |
| Miss Rate (FNR) (%)             |                            2.5    |                                0      |                               0      |
| False Discovery Rate (%)        |                           18.75   |                                2.44   |                               2.44   |
| Negative Pred Value (NPV) (%)   |                           96.87   |                              100      |                             100      |
| ROC-AUC                         |                            0.8844 |                                0.9969 |                               1      |
| PR-AUC (Avg Precision)          |                            0.8944 |                                0.9967 |                               1      |
| Brier Calibration Loss          |                            0.1236 |                                0.0249 |                               0.0153 |
| Cross-Entropy Log-Loss          |                            0      |                                0      |                               0      |
| Inference Latency (ms)          |                            5.958  |                                0.001  |                               6.007  |
| Model Parameters                |                        24676      |                               34      |                          135235      |

## Table 2: Independent Per-Dataset Performance Benchmark

| Performance Evaluation Metric   |   Fast-Accurate ISCAS HT Suite (49 circuits) |   Trust-Hub AES Flagship Suite (30 circuits) |   Overall Multi-Modal Suite (80 circuits) |
|:--------------------------------|---------------------------------------------:|---------------------------------------------:|------------------------------------------:|
| Accuracy (%)                    |                                     100      |                                     100      |                                   98.75   |
| Balanced Accuracy (%)           |                                     100      |                                     100      |                                   98.75   |
| Precision (PPV) (%)             |                                     100      |                                     100      |                                   97.56   |
| Recall / Sensitivity (%)        |                                     100      |                                     100      |                                  100      |
| Specificity (TNR) (%)           |                                     100      |                                     100      |                                   97.5    |
| F1-Score                        |                                       1      |                                       1      |                                    0.9877 |
| Matthews Corr (MCC)             |                                       1      |                                       1      |                                    0.9753 |
| Cohen's Kappa                   |                                       1      |                                       1      |                                    0.975  |
| Youden's J-Index                |                                       1      |                                       1      |                                    0.975  |
| False Alarm Rate (FAR) (%)      |                                       0      |                                       0      |                                    2.5    |
| Miss Rate (FNR) (%)             |                                       0      |                                       0      |                                    0      |
| False Discovery Rate (%)        |                                       0      |                                       0      |                                    2.44   |
| Negative Pred Value (NPV) (%)   |                                     100      |                                     100      |                                  100      |
| ROC-AUC                         |                                       1      |                                       1      |                                    1      |
| PR-AUC (Avg Precision)          |                                       1      |                                       1      |                                    1      |
| Brier Calibration Loss          |                                       0.0126 |                                       0.0064 |                                    0.0153 |
| Cross-Entropy Log-Loss          |                                       0      |                                       0      |                                    0      |
| Inference Latency (ms)          |                                       6.081  |                                       1.877  |                                    6.007  |
| Model Parameters                |                                  135235      |                                  135235      |                               135235      |

## Publication-Ready LaTeX Code for Table 1

```latex
\begin{table*}[htbp]
\centering
\caption{Comprehensive 18-metric performance evaluation of single-modal baselines vs. our proposed Multi-Modal Gated Fusion framework.}
\label{tab:master_framework_metrics}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{Baseline 1 (Graph GNN)} & \textbf{Baseline 2 (Side-Channel)} & \textbf{Proposed (Multi-Modal Fusion)} \\
\midrule
Accuracy (%) & 87.5 & 98.75 & \textbf{98.75} \\
Balanced Accuracy (%) & 87.5 & 98.75 & \textbf{98.75} \\
Precision (PPV) (%) & 81.25 & 97.56 & \textbf{97.56} \\
Recall / Sensitivity (%) & 97.5 & 100.0 & \textbf{100.0} \\
Specificity (TNR) (%) & 77.5 & 97.5 & \textbf{97.5} \\
F1-Score & 0.8864 & 0.9877 & \textbf{0.9877} \\
Matthews Corr (MCC) & 0.7655 & 0.9753 & \textbf{0.9753} \\
Cohen's Kappa & 0.75 & 0.975 & \textbf{0.975} \\
Youden's J-Index & 0.75 & 0.975 & \textbf{0.975} \\
False Alarm Rate (FAR) (%) & 22.5 & 2.5 & \textbf{2.5} \\
Miss Rate (FNR) (%) & 2.5 & 0.0 & \textbf{0.0} \\
False Discovery Rate (%) & 18.75 & 2.44 & \textbf{2.44} \\
Negative Pred Value (NPV) (%) & 96.87 & 100.0 & \textbf{100.0} \\
ROC-AUC & 0.8844 & 0.9969 & \textbf{1.0} \\
PR-AUC (Avg Precision) & 0.8944 & 0.9967 & \textbf{1.0} \\
Brier Calibration Loss & 0.1236 & 0.0249 & \textbf{0.0153} \\
Cross-Entropy Log-Loss & 0.0 & 0.0 & \textbf{0.0} \\
Inference Latency (ms) & 5.958 & 0.001 & \textbf{6.007} \\
Model Parameters & 24676.0 & 34.0 & \textbf{135235.0} \\
\bottomrule
\end{tabular}
\end{table*}
```

## Publication-Ready LaTeX Code for Table 2

```latex
\begin{table*}[htbp]
\centering
\caption{Dataset-wise performance breakdown of the Proposed Multi-Modal Framework across distinct benchmark suites.}
\label{tab:dataset_breakdown_metrics}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{ISCAS HT Suite} & \textbf{Trust-Hub AES Suite} & \textbf{Overall Benchmark} \\
\midrule
Accuracy (%) & 100.0 & 100.0 & \textbf{98.75} \\
Balanced Accuracy (%) & 100.0 & 100.0 & \textbf{98.75} \\
Precision (PPV) (%) & 100.0 & 100.0 & \textbf{97.56} \\
Recall / Sensitivity (%) & 100.0 & 100.0 & \textbf{100.0} \\
Specificity (TNR) (%) & 100.0 & 100.0 & \textbf{97.5} \\
F1-Score & 1.0 & 1.0 & \textbf{0.9877} \\
Matthews Corr (MCC) & 1.0 & 1.0 & \textbf{0.9753} \\
Cohen's Kappa & 1.0 & 1.0 & \textbf{0.975} \\
Youden's J-Index & 1.0 & 1.0 & \textbf{0.975} \\
False Alarm Rate (FAR) (%) & 0.0 & 0.0 & \textbf{2.5} \\
Miss Rate (FNR) (%) & 0.0 & 0.0 & \textbf{0.0} \\
False Discovery Rate (%) & 0.0 & 0.0 & \textbf{2.44} \\
Negative Pred Value (NPV) (%) & 100.0 & 100.0 & \textbf{100.0} \\
ROC-AUC & 1.0 & 1.0 & \textbf{1.0} \\
PR-AUC (Avg Precision) & 1.0 & 1.0 & \textbf{1.0} \\
Brier Calibration Loss & 0.0126 & 0.0064 & \textbf{0.0153} \\
Cross-Entropy Log-Loss & 0.0 & 0.0 & \textbf{0.0} \\
Inference Latency (ms) & 6.081 & 1.877 & \textbf{6.007} \\
Model Parameters & 135235.0 & 135235.0 & \textbf{135235.0} \\
\bottomrule
\end{tabular}
\end{table*}
```
