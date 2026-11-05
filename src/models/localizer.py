"""
localizer.py: Fine-Grained Gate-Level Hardware Trojan Localization.

Integrates GNN node classification, XAI saliency scores, and 1st/2nd Nearest-Neighbour (NN)
topological context refinement to isolate malicious trigger and payload gates with high precision.
"""

from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
import torch


class TrojanGateLocalizer:
    """
    Localizes Hardware Trojan gates in netlists using topological nearest-neighbour refinement.
    Overcomes extreme class imbalance by clustering tightly-coupled trigger-comparator subgraphs.
    """

    def __init__(self, gamma: float = 0.65, k_hop: int = 2):
        self.gamma = gamma          # weight for raw node score vs neighborhood score
        self.k_hop = k_hop          # neighborhood radius (1st and 2nd hop neighbors)

    def localize_gates(
        self,
        nx_graph: nx.DiGraph,
        raw_node_scores: np.ndarray,
        percentile_threshold: float = 98.0
    ) -> Dict[str, object]:
        """
        Refines raw node probabilities using 1st/2nd order nearest-neighbor structural propagation.
        """
        nodes = list(nx_graph.nodes())
        num_nodes = len(nodes)
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        # Undirected view for neighborhood context
        G_undir = nx_graph.to_undirected()

        refined_scores = np.zeros(num_nodes, dtype=float)

        for i, node in enumerate(nodes):
            raw_s = raw_node_scores[i]

            # Collect 1-hop and 2-hop neighbors
            neighbors = set()
            for hop in range(1, self.k_hop + 1):
                hop_nbrs = nx.single_source_shortest_path_length(G_undir, node, cutoff=hop)
                neighbors.update(hop_nbrs.keys())
            neighbors.discard(node)

            if neighbors:
                nbr_scores = [raw_node_scores[node_to_idx[nbr]] for nbr in neighbors if nbr in node_to_idx]
                mean_nbr = float(np.mean(nbr_scores)) if nbr_scores else 0.0
            else:
                mean_nbr = 0.0

            # Structural consistency refinement
            refined_scores[i] = self.gamma * raw_s + (1.0 - self.gamma) * mean_nbr

        # Adaptive thresholding
        thresh = float(np.percentile(refined_scores, percentile_threshold))
        suspicious_indices = [i for i, s in enumerate(refined_scores) if s >= thresh]

        # Extract connected components of suspicious nodes (Trojan Clusters)
        suspicious_nodes = [nodes[i] for i in suspicious_indices]
        suspicious_subgraph = nx_graph.subgraph(suspicious_nodes)
        components = list(nx.weakly_connected_components(suspicious_subgraph))

        # Rank clusters by cumulative anomaly energy
        clusters = []
        for comp in components:
            comp_nodes = list(comp)
            comp_score = float(np.mean([refined_scores[node_to_idx[n]] for n in comp_nodes]))
            clusters.append({
                "nodes": comp_nodes,
                "size": len(comp_nodes),
                "cluster_anomaly_score": round(comp_score, 4),
                "gate_types": [nx_graph.nodes[n].get('gate_type', 'UNKNOWN') for n in comp_nodes]
            })

        clusters.sort(key=lambda c: (c["cluster_anomaly_score"], c["size"]), reverse=True)

        return {
            "refined_scores": refined_scores,
            "threshold": thresh,
            "flagged_gate_count": len(suspicious_indices),
            "flagged_gates": [nodes[i] for i in suspicious_indices],
            "trojan_clusters": clusters,
            "primary_attack_cluster": clusters[0] if clusters else None
        }
