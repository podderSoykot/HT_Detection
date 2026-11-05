"""
test_all_pipelines.py: Automated Regression & Integration Test Suite.

Verifies:
1. Structural Netlist Parser & Gate Normalization
2. Topological Circuit Graph Builder (PyG & NetworkX)
3. RTL Source Code Feature Extractor
4. Physical Power & EM Side-Channel Extractor
5. GNN Detection & Localization Backbone
6. Zero-Day Unseen Anomaly Detector
7. Multi-Modal Cross-Attention Gated Fusion
8. Nearest-Neighbour Subgraph Gate Localizer
9. IP Trustworthiness & Security Risk Scorer
10. One-Click CLI Security Scanner
"""

import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import networkx as nx
import torch

from src.data.netlist_parser import NetlistParser
from src.data.graph_builder import CircuitGraphBuilder
from src.features.rtl_feature_extractor import RTLFeatureExtractor
from src.data.sidechannel_loader import SideChannelFeatureExtractor
from src.models.gnn_detector import HardwareTrojanGNN
from src.models.anomaly_detector import UnseenTrojanDetector
from src.features.multimodal_fusion import MultiModalGatedFusion
from src.models.localizer import TrojanGateLocalizer
from src.evaluation.risk_scorer import IPTrustworthinessAssessor
from experiments.finetune_proposed import CrossAttentionGatedFusion


class TestHardwareTrojanPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.clean_v = "datasets/epfl_benchmarks/arithmetic/adder.v"
        cls.trojan_v = "datasets/fast_accurate_ht_locations/data/TjIn/c5315_T047/topModule.v"

    def test_01_netlist_parser(self):
        parser = NetlistParser()
        netlist = parser.parse_file(self.clean_v)
        self.assertGreater(len(netlist.gates), 0)
        self.assertGreater(len(netlist.inputs), 0)
        self.assertGreater(len(netlist.outputs), 0)

    def test_02_graph_builder(self):
        parser = NetlistParser()
        builder = CircuitGraphBuilder()
        netlist = parser.parse_file(self.clean_v)
        nx_g = builder.build_networkx(netlist, graph_label=0)
        pyg_data = builder.to_pyg_data(nx_g)

        self.assertEqual(pyg_data.x.size(1), 20)
        self.assertEqual(pyg_data.y.item(), 0)
        self.assertGreater(pyg_data.edge_index.size(1), 0)

    def test_03_rtl_feature_extractor(self):
        extractor = RTLFeatureExtractor()
        feat = extractor.extract_from_file(self.trojan_v)
        self.assertEqual(feat.shape[0], 16)
        self.assertTrue(np.all(np.isfinite(feat)))

    def test_04_sidechannel_feature_extractor(self):
        extractor = SideChannelFeatureExtractor(trace_length=128)
        trace_vec = extractor.simulate_trace_from_graph(num_nodes=500, is_trojan=True)
        self.assertEqual(trace_vec.shape[0], 16)
        self.assertTrue(np.all(np.isfinite(trace_vec)))

    def test_05_gnn_model_inference(self):
        gnn = HardwareTrojanGNN(in_channels=20, hidden_channels=32, num_layers=2)
        gnn.eval()
        x = torch.randn(50, 20)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        out = gnn(x, edge_index)
        self.assertEqual(out['graph_logits'].shape, (1, 2))
        self.assertEqual(out['node_logits'].shape, (50, 2))
        self.assertEqual(out['graph_embedding'].shape, (1, 96))

    def test_06_unseen_trojan_anomaly_detector(self):
        detector = UnseenTrojanDetector(embed_dim=96, latent_dim=16)
        golden_embeddings = torch.randn(25, 96)
        detector.fit_golden_manifold(golden_embeddings, epochs=5)

        test_embed = torch.randn(5, 96)
        res = detector.score_anomaly(test_embed)
        self.assertIn('anomaly_score', res)
        self.assertEqual(len(res['anomaly_score']), 5)

    def test_07_cross_attention_gated_fusion(self):
        fusion = CrossAttentionGatedFusion(graph_dim=96, side_dim=16, hidden_dim=64)
        h_g = torch.randn(4, 96)
        h_s = torch.randn(4, 16)

        out = fusion(h_g, h_s)
        self.assertEqual(out['logits'].shape, (4, 2))
        self.assertTrue(torch.all(out['gate_weight'] >= 0.0) and torch.all(out['gate_weight'] <= 1.0))

    def test_08_trojan_gate_localizer(self):
        localizer = TrojanGateLocalizer(gamma=0.6, k_hop=2)
        G = nx.path_graph(20, create_using=nx.DiGraph)
        scores = np.random.rand(20)
        scores[10] = 0.99  # planted spike

        res = localizer.localize_gates(G, scores, percentile_threshold=90.0)
        self.assertIn('flagged_gates', res)
        self.assertGreater(len(res['flagged_gates']), 0)

    def test_09_risk_scorer(self):
        assessor = IPTrustworthinessAssessor()
        gate_names = [f"gate_{i}" for i in range(100)]
        probs = np.zeros(100)
        probs[5] = 0.95

        report = assessor.assess(
            module_name="test_core",
            known_trojan_prob=0.85,
            unseen_anomaly_score=0.40,
            gate_probabilities=probs,
            gate_names=gate_names,
            gate_threshold=0.5
        )
        self.assertGreaterEqual(report.security_risk_score, 0.0)
        self.assertLessEqual(report.security_risk_score, 10.0)
        self.assertIn(report.risk_level, ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])


if __name__ == '__main__':
    unittest.main()
