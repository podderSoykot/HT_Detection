"""
run_pipeline_demo.py: End-to-End Execution of the Multi-Modal Hardware Trojan Pipeline.

Executes:
1. Third-Party IP Netlist Parsing
2. Graph Representation & Feature Encoding
3. GNN-based Detection (Graph classification) & Localization (Node classification)
4. Unseen Trojan Anomaly Scoring
5. IP Trustworthiness Assessment & Security Risk Score Report
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from src.data.dataset_loader import HardwareTrojanDataset
from src.models.gnn_detector import HardwareTrojanGNN
from src.models.anomaly_detector import UnseenTrojanDetector
from src.evaluation.risk_scorer import IPTrustworthinessAssessor


def main():
    print("=" * 70)
    print("  MULTI-MODAL HARDWARE TROJAN DETECTION & RISK ASSESSMENT PIPELINE")
    print("=" * 70)

    # 1. Load Third-Party IP (Candidate Circuit)
    dataset = HardwareTrojanDataset()
    ip_path = "datasets/epfl_benchmarks/arithmetic/adder.v"
    print(f"\n[Step 1] Ingesting Third-Party IP: {ip_path}")
    graph_data = dataset.load_single_verilog(ip_path, label=0)

    if not graph_data:
        print("Failed to load circuit netlist.")
        return

    print(f"  -> Extracted Circuit: {graph_data.module_name}")
    print(f"  -> Total Gates & Ports: {graph_data.num_nodes}")
    print(f"  -> Total Wire Interconnects: {graph_data.edge_index.size(1)}")
    print(f"  -> Feature Encoding Dimension: {graph_data.x.size(1)}")

    # 2. Initialize Deep Learning Model (GNN)
    print("\n[Step 2] Initializing GNN Detection & Localization Model...")
    gnn = HardwareTrojanGNN(
        in_channels=graph_data.x.size(1),
        hidden_channels=64,
        num_layers=3,
        backbone='gcn',
        dropout=0.1
    )
    gnn.eval()

    # 3. Known Trojan Detection & Gate Localization
    print("\n[Step 3] Running GNN Inference...")
    with torch.no_grad():
        out = gnn(graph_data.x, graph_data.edge_index)
        probs = torch.softmax(out['graph_logits'], dim=-1)
        known_trojan_prob = probs[0, 1].item()

        node_probs = torch.softmax(out['node_logits'], dim=-1)[:, 1].numpy()
        circuit_embedding = out['graph_embedding']

    print(f"  -> Known Trojan Probability: {known_trojan_prob:.4f}")

    # 4. Unseen / Zero-Day Trojan Detection (Anomaly Scoring)
    print("\n[Step 4] Evaluating Unseen / Zero-Day Trojan Anomaly...")
    embed_dim = circuit_embedding.size(1)
    unseen_detector = UnseenTrojanDetector(embed_dim=embed_dim)

    # Fit manifold using baseline golden embeddings (synthesized reference manifold)
    ref_golden_embeddings = circuit_embedding.repeat(20, 1) + torch.randn(20, embed_dim) * 0.05
    unseen_detector.fit_golden_manifold(ref_golden_embeddings, epochs=15)

    anomaly_res = unseen_detector.score_anomaly(circuit_embedding)
    unseen_score = float(anomaly_res['anomaly_score'][0])
    print(f"  -> Unseen Trojan Anomaly Score: {unseen_score:.4f}")

    # 5. IP Trustworthiness & Security Risk Assessment
    print("\n[Step 5] Performing IP Trustworthiness & Risk Score Assessment...")
    assessor = IPTrustworthinessAssessor()
    report = assessor.assess(
        module_name=graph_data.module_name,
        known_trojan_prob=known_trojan_prob,
        unseen_anomaly_score=unseen_score,
        gate_probabilities=node_probs,
        gate_names=graph_data.node_names,
        gate_threshold=0.5
    )

    # 6. Display Audit Report
    print("\n" + "=" * 70)
    print("                      SECURITY AUDIT REPORT")
    print("=" * 70)
    print(f" Circuit Under Test (IP) : {report.module_name}")
    print(f" Security Risk Score     : {report.security_risk_score} / 10.0")
    print(f" Threat Level            : [{report.risk_level}]")
    print(f" Trustworthiness Index   : {report.trustworthiness_pct}%")
    print(f" Known Trojan Confidence : {report.known_trojan_prob * 100:.2f}%")
    print(f" Zero-Day Anomaly Score  : {report.unseen_anomaly_score * 100:.2f}%")
    print(f" Suspicious Gates Count  : {report.suspicious_gate_count} / {len(graph_data.node_names)}")
    print(f" Final Audit Verdict     : {report.verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
