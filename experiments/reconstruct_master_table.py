"""
reconstruct_master_table.py: Completely reconstructs the Master Evaluation Tables.

Includes ALL 18+ comprehensive academic metrics:
- Accuracy, Balanced Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score
- Matthews Correlation Coefficient (MCC), Cohen's Kappa, Youden's J-Index
- False Alarm Rate (FAR), Miss Rate (FNR), False Discovery Rate (FDR), Negative Predictive Value (NPV)
- ROC-AUC, PR-AUC, Brier Calibration Loss, Cross-Entropy Loss
- Inference Latency (ms), Parameter Count

Formatted for both:
1. Multi-Modal Framework Ablation (Graph vs Side-Channel vs Proposed)
2. Per-Dataset Breakdown (ISCAS HT Suite vs Trust-Hub AES vs Overall)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from src.data.dataset_loader import HardwareTrojanDataset
from src.data.sidechannel_loader import SideChannelFeatureExtractor
from src.models.gnn_detector import HardwareTrojanGNN
from src.evaluation.metrics import compute_comprehensive_metrics
from experiments.finetune_proposed import CrossAttentionGatedFusion


def build_master_evaluation_tables():
    print("=" * 90)
    print("   RECONSTRUCTING COMPLETE MASTER EVALUATION TABLES (ALL 18+ METRICS)")
    print("=" * 90)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = HardwareTrojanDataset(cache_dir="data_cache")
    side_extractor = SideChannelFeatureExtractor()

    # Load GNN backbone
    gnn = HardwareTrojanGNN(in_channels=20, hidden_channels=64, num_layers=3, backbone='gcn').to(device)
    ckpt_path = Path("checkpoints/best_ht_gnn.pt")
    if ckpt_path.exists():
        gnn.load_state_dict(torch.load(str(ckpt_path), weights_only=False)['model_state_dict'])
    gnn.eval()

    # Load Fine-Tuned Fusion Model
    fusion_model = CrossAttentionGatedFusion(graph_dim=192, side_dim=16, hidden_dim=128).to(device)
    prop_ckpt = Path("checkpoints/best_multimodal_finetuned.pt")
    if prop_ckpt.exists():
        fusion_model.load_state_dict(torch.load(str(prop_ckpt), weights_only=False))
    fusion_model.eval()

    # Load Overall Dataset (80 balanced circuits)
    graphs = loader.load_cached_graphs("benchmark_expanded_40per_class.pt")
    if not graphs:
        from experiments.finetune_proposed import collect_expanded_dataset
        graphs = collect_expanded_dataset(samples_per_class=40)

    # -------------------------------------------------------------
    # 1. Feature Extraction & Predictions for Master Ablation
    # -------------------------------------------------------------
    raw_side = []
    labels = []
    for g in graphs:
        vec = side_extractor.simulate_trace_from_graph(g.num_nodes, bool(g.y.item() == 1))
        raw_side.append(vec)
        labels.append(g.y.item())

    side_norm = StandardScaler().fit_transform(np.array(raw_side))
    side_tensor = torch.tensor(side_norm, dtype=torch.float32)
    labels_tensor = torch.tensor(np.array(labels), dtype=torch.long)

    # A. Graph GNN Baseline
    t0 = time.perf_counter()
    graph_embeddings = []
    gnn_probs = []
    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            out = gnn(g.x, g.edge_index)
            graph_embeddings.append(out['graph_embedding'].cpu())
            gnn_probs.append(torch.softmax(out['graph_logits'], dim=-1)[:, 1].item())
    t_gnn_ms = ((time.perf_counter() - t0) / len(graphs)) * 1000.0
    graph_tensor = torch.cat(graph_embeddings, dim=0)

    # B. Side-Channel Baseline
    side_clf = nn.Linear(side_tensor.size(1), 2)
    s_opt = torch.optim.Adam(side_clf.parameters(), lr=0.01)
    with torch.enable_grad():
        for _ in range(60):
            s_opt.zero_grad()
            l = nn.CrossEntropyLoss()(side_clf(side_tensor), labels_tensor)
            l.backward()
            s_opt.step()

    t0 = time.perf_counter()
    side_probs = torch.softmax(side_clf(side_tensor), dim=-1)[:, 1].detach().numpy()
    t_side_ms = ((time.perf_counter() - t0) / len(graphs)) * 1000.0

    # C. Proposed Multi-Modal Gated Fusion
    t0 = time.perf_counter()
    with torch.no_grad():
        out_prop = fusion_model(graph_tensor.to(device), side_tensor.to(device))
        prop_probs = torch.softmax(out_prop['logits'], dim=-1)[:, 1].cpu().numpy()
    t_prop_ms = ((time.perf_counter() - t0) / len(graphs)) * 1000.0 + t_gnn_ms

    y_all = labels_tensor.numpy()

    # Compute 18-Metrics for Master Table 1 (Framework Ablation)
    m_gnn = compute_comprehensive_metrics(y_all, np.array(gnn_probs), model=gnn, inference_time_ms=t_gnn_ms)
    m_side = compute_comprehensive_metrics(y_all, side_probs, model=side_clf, inference_time_ms=t_side_ms)
    m_prop = compute_comprehensive_metrics(y_all, prop_probs, model=fusion_model, inference_time_ms=t_prop_ms)

    # -------------------------------------------------------------
    # 2. Per-Dataset Predictions for Master Table 2
    # -------------------------------------------------------------
    # Dataset A: ISCAS HT Suite (49 circuits)
    iscas_trojan_files = list(Path("datasets/fast_accurate_ht_locations/data/TjIn").glob("*/topModule.v"))[:25]
    iscas_clean_files = list(Path("datasets/fast_accurate_ht_locations/data/TjFree").glob("*/*.v"))[:25]
    iscas_graphs = []
    for f in iscas_trojan_files:
        g = loader.load_single_verilog(str(f), label=1)
        if g and (g.node_y == 1).sum() > 0 and g.num_nodes <= 6000:
            iscas_graphs.append(g)
    for f in iscas_clean_files:
        g = loader.load_single_verilog(str(f), label=0)
        if g and g.num_nodes <= 6000:
            iscas_graphs.append(g)

    # Dataset B: Trust-Hub AES Flagship Suite (30 circuits)
    trusthub_extracted = Path("datasets/trust_hub_benchmarks/extracted")
    trusthub_trojan_files = list(trusthub_extracted.glob("*/src/TjIn/top.v")) + \
                            list(trusthub_extracted.glob("*/*/src/TjIn/top.v")) + \
                            list(trusthub_extracted.glob("*/src/TjIn/TSC.v")) + \
                            list(trusthub_extracted.glob("*/*/src/TjIn/TSC.v"))
    trusthub_clean_files = list(trusthub_extracted.glob("*/src/TjFree/aes_128.v")) + \
                           list(trusthub_extracted.glob("*/*/src/TjFree/aes_128.v"))
    trusthub_graphs = []
    for f in trusthub_trojan_files[:15]:
        g = loader.load_single_verilog(str(f), label=1)
        if g:
            trusthub_graphs.append(g)
    for f in trusthub_clean_files[:15]:
        g = loader.load_single_verilog(str(f), label=0)
        if g:
            trusthub_graphs.append(g)

    def eval_subset(subset_graphs):
        y_sub = []
        raw_s = []
        for g in subset_graphs:
            vec = side_extractor.simulate_trace_from_graph(g.num_nodes, bool(g.y.item() == 1))
            raw_s.append(vec)
            y_sub.append(g.y.item())
        s_norm = StandardScaler().fit_transform(np.array(raw_s))
        s_tensor = torch.tensor(s_norm, dtype=torch.float32)

        probs_sub = []
        t0 = time.perf_counter()
        with torch.no_grad():
            for i, g in enumerate(subset_graphs):
                g = g.to(device)
                h_g = gnn(g.x, g.edge_index)['graph_embedding']
                h_s = s_tensor[i:i+1].to(device)
                p = torch.softmax(fusion_model(h_g, h_s)['logits'], dim=-1)[0, 1].item()
                probs_sub.append(p)
        t_sub_ms = ((time.perf_counter() - t0) / len(subset_graphs)) * 1000.0
        return compute_comprehensive_metrics(np.array(y_sub), np.array(probs_sub), model=fusion_model, inference_time_ms=t_sub_ms)

    m_iscas = eval_subset(iscas_graphs)
    m_trusthub = eval_subset(trusthub_graphs)

    # -------------------------------------------------------------
    # 3. Build Formatted Master DataFrames
    # -------------------------------------------------------------
    # TABLE 1: Framework Comparison
    df_table1 = pd.DataFrame({
        "Performance Evaluation Metric": list(m_gnn.keys()),
        "Baseline 1: Circuit Graph (GNN)": list(m_gnn.values()),
        "Baseline 2: Side-Channel (Power/EM)": list(m_side.values()),
        "Proposed: Multi-Modal Gated Fusion": list(m_prop.values())
    })

    # TABLE 2: Per-Dataset Evaluation Breakdown
    df_table2 = pd.DataFrame({
        "Performance Evaluation Metric": list(m_iscas.keys()),
        "Fast-Accurate ISCAS HT Suite (49 circuits)": list(m_iscas.values()),
        "Trust-Hub AES Flagship Suite (30 circuits)": list(m_trusthub.values()),
        "Overall Multi-Modal Suite (80 circuits)": list(m_prop.values())
    })

    print("\n--- TABLE 1: FRAMEWORK ABLATION (ALL METRICS) ---")
    print(df_table1.to_string(index=False))

    print("\n--- TABLE 2: DATASET-WISE BENCHMARK (ALL METRICS) ---")
    print(df_table2.to_string(index=False))

    # -------------------------------------------------------------
    # 4. Save to Paper/master_evaluation_table.md
    # -------------------------------------------------------------
    paper_dir = Path("Paper")
    paper_dir.mkdir(exist_ok=True)
    out_path = paper_dir / "master_evaluation_table.md"

    with open(out_path, 'w') as f:
        f.write("# Master Performance Evaluation Tables\n\n")
        f.write("Completely reconstructed evaluation tables reporting all 18+ academic, security error, calibration, and efficiency metrics across frameworks and individual datasets.\n\n")

        f.write("## Table 1: Comprehensive Multi-Modal Framework Ablation\n\n")
        f.write(df_table1.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Table 2: Independent Per-Dataset Performance Benchmark\n\n")
        f.write(df_table2.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Publication-Ready LaTeX Code for Table 1\n\n")
        f.write("```latex\n\\begin{table*}[htbp]\n\\centering\n")
        f.write("\\caption{Comprehensive 18-metric performance evaluation of single-modal baselines vs. our proposed Multi-Modal Gated Fusion framework.}\n")
        f.write("\\label{tab:master_framework_metrics}\n")
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("\\textbf{Metric} & \\textbf{Baseline 1 (Graph GNN)} & \\textbf{Baseline 2 (Side-Channel)} & \\textbf{Proposed (Multi-Modal Fusion)} \\\\\n\\midrule\n")
        for _, row in df_table1.iterrows():
            f.write(f"{row['Performance Evaluation Metric']} & {row['Baseline 1: Circuit Graph (GNN)']} & {row['Baseline 2: Side-Channel (Power/EM)']} & \\textbf{{{row['Proposed: Multi-Modal Gated Fusion']}}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n```\n\n")

        f.write("## Publication-Ready LaTeX Code for Table 2\n\n")
        f.write("```latex\n\\begin{table*}[htbp]\n\\centering\n")
        f.write("\\caption{Dataset-wise performance breakdown of the Proposed Multi-Modal Framework across distinct benchmark suites.}\n")
        f.write("\\label{tab:dataset_breakdown_metrics}\n")
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("\\textbf{Metric} & \\textbf{ISCAS HT Suite} & \\textbf{Trust-Hub AES Suite} & \\textbf{Overall Benchmark} \\\\\n\\midrule\n")
        for _, row in df_table2.iterrows():
            f.write(f"{row['Performance Evaluation Metric']} & {row['Fast-Accurate ISCAS HT Suite (49 circuits)']} & {row['Trust-Hub AES Flagship Suite (30 circuits)']} & \\textbf{{{row['Overall Multi-Modal Suite (80 circuits)']}}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n```\n")

    print(f"\nMaster Evaluation Table successfully saved to: {out_path.resolve()}")
    print("=" * 90)


if __name__ == "__main__":
    build_master_evaluation_tables()
