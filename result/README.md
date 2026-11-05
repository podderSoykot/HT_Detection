# Hardware Trojan Detection: Comprehensive Research Results Archive

This folder consolidates all empirical benchmarks, evaluation tables, publication-ready figures, and trained deep learning models produced for the Multi-Modal Hardware Trojan Detection project.

---

## Folder Organization

```
result/
├── tables/                   # All evaluation tables (Markdown, LaTeX, and CSV)
│   ├── master_evaluation_table.md       # Master 18-metric tables for Framework Ablation and Dataset-Wise Breakdown
│   ├── table1_framework_ablation.csv    # CSV export of Framework Ablation metrics
│   ├── table2_dataset_wise_metrics.csv  # CSV export of Per-Dataset metrics
│   ├── dataset_wise_breakdown.md        # Independent audit on EPFL, ISCAS, and Trust-Hub AES
│   └── extended_metrics_evaluation.md   # Definitions of all 18 security & efficiency metrics
│
├── plots/                    # High-resolution (300 DPI) publication-grade figures
│   ├── hierarchical_schematic.png       # Left-to-Right logic flow of Trojan trigger tree (c5315_T047)
│   ├── xai_attribution_heatmap.png      # GNN Saliency Attribution Heatmap isolating attack path
│   ├── multi_circuit_comparison_gallery.png # 4-Panel comparison across benchmark families
│   ├── circuit_size_vs_trojan_gates.png # Host gate count vs Trojan gate count
│   ├── trojan_overhead_distribution.png # Trojan stealthiness (% overhead) distribution
│   ├── gate_type_distribution.png       # Gate types utilized in normal logic vs Trojan triggers
│   ├── adder_golden.png                 # EPFL 128-bit Adder (Golden) Graph
│   ├── c3540_t018.png                   # ISCAS-85 c3540 ALU (Trojan) Graph
│   ├── c5315_t047.png                   # ISCAS-85 c5315 9-bit ALU (Trojan) Graph
│   ├── c6288_t067.png                   # ISCAS-85 c6288 16x16 Multiplier (Trojan) Graph
│   └── pic16f84_t100.png                # PIC16F84 Microcontroller Core (Trojan) Graph
│
├── checkpoints/              # Trained PyTorch Model Weights
│   ├── best_multimodal_finetuned.pt     # Best Fine-Tuned Cross-Attention Gated Fusion Model (98.75% Acc, 100% Rec, 1.0 AUC)
│   └── best_ht_gnn.pt                   # Pre-trained Hardware Trojan GNN Backbone
│
└── interactive/              # Interactive Web Visualizers
    └── circuit_graph_interactive.html   # Live HTML/SVG schematic viewer with gate inspection panel
```

---

## Key Experimental Results Summary

- **Overall Detection Accuracy**: **98.75%**
- **Trojan Recall / Sensitivity**: **100.00%** (Zero False Negatives)
- **False Alarm Rate (FAR)**: **2.50%**
- **Matthews Correlation Coefficient (MCC)**: **0.9753**
- **ROC-AUC & PR-AUC**: **1.0000**
- **Inference Latency**: **6.01 ms / Circuit**
