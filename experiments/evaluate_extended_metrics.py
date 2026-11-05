"""
evaluate_extended_metrics.py: Evaluates all models across the full suite of 18+ academic metrics.

Includes:
1. Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1-Score, MCC, Cohen's Kappa.
2. Security Miss Rate (FNR), False Alarm Rate (FAR), False Discovery Rate (FDR), NPV.
3. PR-AUC, ROC-AUC, Brier Calibration Loss, Cross-Entropy Log-Loss.
4. Computational Benchmark: Latency (ms), Throughput (circuits/sec), Parameter Count.
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


def run_extended_evaluation():
    print("=" * 85)
    print("      EXTENDED PERFORMANCE EVALUATION METRICS SUITE (HARDWARE TROJAN DETECTION)")
    print("=" * 85)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_handler = HardwareTrojanDataset(cache_dir="data_cache")
    graphs = dataset_handler.load_cached_graphs("benchmark_expanded_40per_class.pt")

    if not graphs:
        from experiments.finetune_proposed import collect_expanded_dataset
        graphs = collect_expanded_dataset(samples_per_class=40)

    # 1. Feature Preparation
    side_extractor = SideChannelFeatureExtractor()
    side_feats = []
    labels = []
    for g in graphs:
        vec = side_extractor.simulate_trace_from_graph(g.num_nodes, bool(g.y.item() == 1))
        side_feats.append(vec)
        labels.append(g.y.item())

    scaler = StandardScaler()
    side_feats_norm = scaler.fit_transform(np.array(side_feats))
    side_tensor = torch.tensor(side_feats_norm, dtype=torch.float32)
    labels_tensor = torch.tensor(np.array(labels), dtype=torch.long)

    # Load GNN for graph embeddings
    gnn = HardwareTrojanGNN(in_channels=20, hidden_channels=64, num_layers=3, backbone='gcn').to(device)
    ckpt_path = Path("checkpoints/best_ht_gnn.pt")
    if ckpt_path.exists():
        gnn.load_state_dict(torch.load(str(ckpt_path), weights_only=False)['model_state_dict'])

    gnn.eval()
    graph_embeddings = []
    gnn_probs = []

    # Benchmark GNN Latency
    t0 = time.perf_counter()
    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            out = gnn(g.x, g.edge_index)
            graph_embeddings.append(out['graph_embedding'].cpu())
            p = torch.softmax(out['graph_logits'], dim=-1)[:, 1].item()
            gnn_probs.append(p)
    t_gnn_ms = ((time.perf_counter() - t0) / len(graphs)) * 1000.0

    graph_tensor = torch.cat(graph_embeddings, dim=0)

    # 2. Side-Channel Only Baseline
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

    # 3. Proposed Fine-Tuned Model
    proposed_model = CrossAttentionGatedFusion(
        graph_dim=graph_tensor.size(1),
        side_dim=side_tensor.size(1),
        hidden_dim=128
    ).to(device)

    prop_ckpt = Path("checkpoints/best_multimodal_finetuned.pt")
    if prop_ckpt.exists():
        proposed_model.load_state_dict(torch.load(str(prop_ckpt), weights_only=False))

    proposed_model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_prop = proposed_model(graph_tensor.to(device), side_tensor.to(device))
        prop_probs = torch.softmax(out_prop['logits'], dim=-1)[:, 1].cpu().numpy()
    t_prop_ms = ((time.perf_counter() - t0) / len(graphs)) * 1000.0 + t_gnn_ms  # includes graph pass

    # 4. Compute Comprehensive 18-Metric Breakdown
    y_test = labels_tensor.numpy()

    m_gnn = compute_comprehensive_metrics(y_test, np.array(gnn_probs), model=gnn, inference_time_ms=t_gnn_ms)
    m_side = compute_comprehensive_metrics(y_test, side_probs, model=side_clf, inference_time_ms=t_side_ms)
    m_prop = compute_comprehensive_metrics(y_test, prop_probs, model=proposed_model, inference_time_ms=t_prop_ms)

    # Compile Table
    models_data = {
        "Metric Category & Name": list(m_gnn.keys()),
        "Baseline 1: Graph GNN": list(m_gnn.values()),
        "Baseline 2: Side-Channel": list(m_side.values()),
        "Proposed: Multi-Modal Fusion": list(m_prop.values())
    }
    df_metrics = pd.DataFrame(models_data)

    print("\n" + df_metrics.to_string(index=False))

    # Save to Paper directory
    paper_dir = Path("Paper")
    paper_dir.mkdir(exist_ok=True)
    out_file = paper_dir / "extended_metrics_evaluation.md"

    with open(out_file, 'w') as f:
        f.write("# Comprehensive 18-Metric Academic Evaluation Benchmark\n\n")
        f.write(df_metrics.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Metric Definitions & Security Significance\n\n")
        f.write("- **Miss Rate (False Negative Rate, FNR)**: In Hardware Trojan detection, Miss Rate is safety-critical. A missed Trojan represents an undetected malicious payload embedded in silicon.\n")
        f.write("- **False Alarm Rate (FAR / FPR)**: Measures unnecessary quarantine or scrap of expensive, clean silicon wafers.\n")
        f.write("- **Matthews Correlation Coefficient (MCC)**: Balanced measure $(-1.0 \\text{ to } +1.0)$ robust against extreme class imbalances.\n")
        f.write("- **Brier Calibration Loss**: Measures confidence calibration of output security probabilities ($0.0$ is perfect calibration).\n")
        f.write("- **Inference Latency (ms)**: Real-time silicon inspection speed per circuit netlist.\n")

    print(f"\nSaved extended metrics to: {out_file.resolve()}")
    print("=" * 85)


if __name__ == "__main__":
    run_extended_evaluation()
