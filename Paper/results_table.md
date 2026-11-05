# Experimental Evaluation & Ablation Results (Fine-Tuned)

## 1. Multi-Modal Hardware Trojan Detection Performance

| Framework / Modality                              | Accuracy   | Precision   | Recall   |   F1-Score |   ROC-AUC |
|:--------------------------------------------------|:-----------|:------------|:---------|-----------:|----------:|
| Baseline 1: Circuit Graph Only (GNN)              | 87.5%      | 83.3%       | 83.3%    |     0.8333 |    0.9417 |
| Baseline 2: Side-Channel Only (Power/EM)          | 100.0%     | 100.0%      | 100.0%   |     1      |    1      |
| Proposed: Fine-Tuned Cross-Attention Gated Fusion | 100.0%     | 100.0%      | 100.0%   |     1      |    1      |

- **Optimal Gated Fusion Alpha**: `0.400`
- **Fine-Tuning Convergence**: Reached peak F1 after cosine-annealed optimization.

## 2. Publication LaTeX Table

```latex
\begin{table}[htbp]
\centering
\caption{Benchmarking Single-Modal Baselines vs. Proposed Fine-Tuned Cross-Attention Gated Fusion.}
\label{tab:finetuned_results}
\begin{tabular}{lccccc}
\toprule
\textbf{Framework / Modality} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{AUC} \\
\midrule
Baseline 1: Circuit Graph Only (GNN) & 87.5% & 83.3% & 83.3% & 0.8333 & 0.9417 \\
Baseline 2: Side-Channel Only (Power/EM) & 100.0% & 100.0% & 100.0% & 1.0000 & 1.0000 \\
Proposed: Fine-Tuned Cross-Attention Gated Fusion & 100.0% & 100.0% & 100.0% & 1.0000 & 1.0000 \\
\bottomrule
\end{tabular}
\end{table}
```
