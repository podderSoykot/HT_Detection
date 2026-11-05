"""
scan_ip.py: One-Click Command-Line Security Audit Tool for Third-Party IP (3PIP).

Usage:
    python scan_ip.py --input datasets/epfl_benchmarks/arithmetic/adder.v
    python scan_ip.py --input datasets/fast_accurate_ht_locations/data/TjIn/c5315_T047/topModule.v
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

from src.data.netlist_parser import NetlistParser
from src.data.graph_builder import CircuitGraphBuilder
from src.data.sidechannel_loader import SideChannelFeatureExtractor
from src.models.gnn_detector import HardwareTrojanGNN
from src.models.localizer import TrojanGateLocalizer
from src.explainability.xai_explainer import HardwareTrojanExplainer
from src.evaluation.risk_scorer import IPTrustworthinessAssessor
from experiments.finetune_proposed import CrossAttentionGatedFusion


def scan_circuit(netlist_file: str, out_dir: str = "result/audit_reports"):
    path = Path(netlist_file)
    if not path.exists():
        print(f"Error: File not found: {netlist_file}")
        sys.exit(1)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 78)
    print("      AI-POWERED HARDWARE TROJAN & 3PIP SECURITY AUDIT SCANNER")
    print("=" * 78)
    print(f"  Target Circuit Netlist : {path.resolve()}")
    print(f"  Execution Device       : {device.type.upper()}")

    # 1. Parsing Netlist
    print("\n[1/5] Ingesting and Parsing Gate-Level Structural Netlist...")
    parser = NetlistParser()
    builder = CircuitGraphBuilder()
    netlist = parser.parse_file(str(path))

    if not netlist.gates:
        print("  Warning: No gates parsed. Ensure file is a structural gate-level netlist.")
        sys.exit(1)

    nx_g = builder.build_networkx(netlist)
    pyg_data = builder.to_pyg_data(nx_g).to(device)

    print(f"  -> Module Name          : {netlist.module_name}")
    print(f"  -> Total Graph Nodes    : {pyg_data.num_nodes} (Gates & Ports)")
    print(f"  -> Interconnect Wires   : {pyg_data.edge_index.size(1)}")
    print(f"  -> Primary Inputs (PI)  : {len(netlist.inputs)}")
    print(f"  -> Primary Outputs (PO) : {len(netlist.outputs)}")

    # 2. Loading AI Models
    print("\n[2/5] Loading Pre-Trained Multi-Modal AI Detection Engines...")
    gnn = HardwareTrojanGNN(in_channels=20, hidden_channels=64, num_layers=3, backbone='gcn').to(device)
    ckpt_gnn = Path("checkpoints/best_ht_gnn.pt")
    if ckpt_gnn.exists():
        gnn.load_state_dict(torch.load(str(ckpt_gnn), weights_only=False)['model_state_dict'])
    gnn.eval()

    fusion_model = CrossAttentionGatedFusion(graph_dim=192, side_dim=16, hidden_dim=128).to(device)
    ckpt_fusion = Path("checkpoints/best_multimodal_finetuned.pt")
    if ckpt_fusion.exists():
        fusion_model.load_state_dict(torch.load(str(ckpt_fusion), weights_only=False))
    fusion_model.eval()

    # 3. Multi-Modal Inference
    print("\n[3/5] Running Multi-Modal Inference & Cross-Attention Fusion...")
    with torch.no_grad():
        gnn_out = gnn(pyg_data.x, pyg_data.edge_index)
        h_graph = gnn_out['graph_embedding']

        # Extract/Simulate Physical Side-Channel Vector
        side_extractor = SideChannelFeatureExtractor()
        side_vec = side_extractor.simulate_trace_from_graph(pyg_data.num_nodes, is_trojan=False)
        h_side = torch.tensor(side_vec, dtype=torch.float32).unsqueeze(0).to(device)

        # Fused Decision
        fused_out = fusion_model(h_graph, h_side)
        probs = torch.softmax(fused_out['logits'], dim=-1)
        trojan_prob = float(probs[0, 1].item())
        alpha = float(fused_out['gate_weight'].item())

    # 4. Explainability & Gate-Level Localization
    print("\n[4/5] Executing XAI Saliency Tracing & Nearest-Neighbour Gate Isolation...")
    explainer = HardwareTrojanExplainer(gnn)
    xai_res = explainer.explain_circuit(pyg_data, target_class=1, top_k_gates=10)

    localizer = TrojanGateLocalizer(gamma=0.65, k_hop=2)
    loc_res = localizer.localize_gates(nx_g, xai_res['all_attributions'], percentile_threshold=98.0)

    # Save XAI Heatmap
    heatmap_file = out_path / f"{path.stem}_xai_heatmap.png"
    explainer.generate_xai_heatmap(nx_g, xai_res['all_attributions'], str(heatmap_file))

    # 5. Security Risk Assessment & Trustworthiness
    print("\n[5/5] Synthesizing IP Trustworthiness & Security Risk Score...")
    assessor = IPTrustworthinessAssessor()
    report = assessor.assess(
        module_name=netlist.module_name,
        known_trojan_prob=trojan_prob,
        unseen_anomaly_score=float(np.mean(xai_res['all_attributions'])),
        gate_probabilities=loc_res['refined_scores'],
        gate_names=pyg_data.node_names,
        gate_threshold=loc_res['threshold']
    )

    # Display Report
    print("\n" + "=" * 78)
    print("                     SECURITY AUDIT SCAN RESULTS")
    print("=" * 78)
    print(f" Target Module Under Test : {report.module_name}")
    print(f" Security Risk Score      : {report.security_risk_score} / 10.0")
    print(f" Threat Severity Level    : [{report.risk_level}]")
    print(f" IP Trustworthiness Index : {report.trustworthiness_pct}%")
    print(f" Trojan Detection Prob    : {report.known_trojan_prob * 100:.2f}%")
    print(f" Cross-Attention Weight   : {alpha:.3f} (Graph) vs {1.0-alpha:.3f} (Side-Channel)")
    print(f" Flagged Suspicious Gates : {report.suspicious_gate_count} / {pyg_data.num_nodes}")
    if report.top_suspicious_gates:
        print(f" Top Suspicious Gates     : {', '.join(report.top_suspicious_gates[:5])}")
    print(f" Final Audit Verdict      : {report.verdict}")
    print(f" XAI Heatmap Saved        : {heatmap_file.resolve()}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Hardware Trojan Security Scanner")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to structural Verilog netlist file")
    parser.add_argument("--out-dir", "-o", type=str, default="result/audit_reports", help="Directory for output reports")
    args = parser.parse_args()
    scan_circuit(args.input, args.out_dir)
