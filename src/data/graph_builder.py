"""
graph_builder.py: Converts CircuitNetlist into NetworkX DiGraph and PyTorch Geometric Data.

Implements Gate-Level Netlist Graph modeling where:
- Nodes = Logic Gates, Primary Inputs (PI), Primary Outputs (PO)
- Directed Edges = Signal flow across interconnecting wires
- Node Features = One-hot gate type, fan-in, fan-out, logic depth, sequential flag
- Labels = Graph-level (Trojan presence) & Node-level (Trojan localization)
"""

from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from src.data.netlist_parser import CircuitNetlist, GateInstance


# Canonical vocabulary of gate types for one-hot encoding
GATE_VOCABULARY = [
    'PI',       # Primary Input
    'PO',       # Primary Output
    'AND',
    'NAND',
    'OR',
    'NOR',
    'XOR',
    'XNOR',
    'NOT',
    'BUF',
    'DFF',
    'LATCH',
    'MUX',
    'AOI',
    'OAI',
    'OTHER'
]
GATE_TO_IDX: Dict[str, int] = {gate: i for i, gate in enumerate(GATE_VOCABULARY)}


class CircuitGraphBuilder:
    """Builds topological NetworkX and PyG graphs from parsed circuit netlists."""

    def __init__(self, gate_vocab: Optional[List[str]] = None):
        self.vocab = gate_vocab or GATE_VOCABULARY
        self.gate_to_idx = {g: i for i, g in enumerate(self.vocab)}
        self.num_classes = len(self.vocab)

    def build_networkx(self, netlist: CircuitNetlist, graph_label: int = 0) -> nx.DiGraph:
        """
        Converts a CircuitNetlist into a directed graph (nx.DiGraph).
        Nodes: Primary Inputs, Primary Outputs, and Logic Gates.
        Edges: Signal wire routing from source driver to consumer.
        """
        G = nx.DiGraph()
        G.graph['module_name'] = netlist.module_name
        G.graph['label'] = graph_label
        G.graph['file_path'] = netlist.file_path or ""

        # Map wire names to the producing driver node name
        wire_driver: Dict[str, str] = {}

        # 1. Add Primary Inputs
        for pi_name in netlist.inputs:
            node_id = f"PI_{pi_name}"
            G.add_node(
                node_id,
                gate_type='PI',
                gate_type_idx=self.gate_to_idx['PI'],
                is_sequential=False,
                is_trojan=False,
                wire_name=pi_name
            )
            wire_driver[pi_name] = node_id

        # 2. Add Logic Gates
        for gate in netlist.gates:
            node_id = gate.inst_name
            gtype = gate.gate_type if gate.gate_type in self.gate_to_idx else 'OTHER'
            G.add_node(
                node_id,
                gate_type=gtype,
                gate_type_idx=self.gate_to_idx[gtype],
                is_sequential=gate.is_sequential,
                is_trojan=gate.is_trojan,
                raw_type=gate.raw_type
            )
            for out_w in gate.outputs:
                wire_driver[out_w] = node_id

        # 3. Add Primary Outputs
        for po_name in netlist.outputs:
            node_id = f"PO_{po_name}"
            G.add_node(
                node_id,
                gate_type='PO',
                gate_type_idx=self.gate_to_idx['PO'],
                is_sequential=False,
                is_trojan=False,
                wire_name=po_name
            )
            if po_name in wire_driver:
                G.add_edge(wire_driver[po_name], node_id, wire=po_name)

        # 4. Connect Gate inputs from wire drivers
        for gate in netlist.gates:
            sink_id = gate.inst_name
            for in_w in gate.inputs:
                if in_w in wire_driver:
                    src_id = wire_driver[in_w]
                    if src_id != sink_id:  # avoid direct self-loops in pure combinational gates
                        G.add_edge(src_id, sink_id, wire=in_w)

        # 5. Compute topological depth (Logic Depth from PIs)
        self._compute_logic_depth(G)

        return G

    def _compute_logic_depth(self, G: nx.DiGraph):
        """Computes shortest and longest path logic depth from Primary Inputs."""
        # Initialize depth to 0 for PI, -1 for others
        depths = {n: 0 if G.nodes[n].get('gate_type') == 'PI' else -1 for n in G.nodes()}

        try:
            # If DAG (pure combinational or loop-free unfolded graph)
            topo_order = list(nx.topological_sort(G))
            for n in topo_order:
                preds = list(G.predecessors(n))
                if preds:
                    max_pred_depth = max(depths[p] for p in preds)
                    depths[n] = max_pred_depth + 1
        except nx.NetworkXUnfeasible:
            # Graph has sequential cycles/feedback loops; use multi-source BFS
            pi_sources = [n for n in G.nodes() if G.nodes[n].get('gate_type') == 'PI']
            for src in pi_sources:
                lengths = nx.single_source_shortest_path_length(G, src)
                for target, length in lengths.items():
                    depths[target] = max(depths[target], length)

        # Assign back to nodes
        for node, d in depths.items():
            G.nodes[node]['depth'] = max(0, d)
            G.nodes[node]['fan_in'] = G.in_degree(node)
            G.nodes[node]['fan_out'] = G.out_degree(node)

    def to_pyg_data(self, G: nx.DiGraph) -> Data:
        """
        Converts a NetworkX circuit graph into a PyTorch Geometric Data object.
        Feature vector per node:
        [One-Hot Gate Type (16-dim), Normalized Fan-In (1), Normalized Fan-Out (1),
         Normalized Depth (1), Is_Sequential (1)] -> 20-dim feature vector.
        """
        nodes = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        num_nodes = len(nodes)

        # Node features
        feature_dim = self.num_classes + 4
        x = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        node_labels = np.zeros(num_nodes, dtype=np.int64)

        for i, n in enumerate(nodes):
            attrs = G.nodes[n]
            # One-hot gate type
            gate_idx = attrs.get('gate_type_idx', self.gate_to_idx['OTHER'])
            x[i, gate_idx] = 1.0

            # Structural attributes
            fan_in = attrs.get('fan_in', 0)
            fan_out = attrs.get('fan_out', 0)
            depth = attrs.get('depth', 0)
            is_seq = 1.0 if attrs.get('is_sequential', False) else 0.0

            # Normalized numerical features
            x[i, self.num_classes] = np.log1p(fan_in)
            x[i, self.num_classes + 1] = np.log1p(fan_out)
            x[i, self.num_classes + 2] = np.log1p(depth)
            x[i, self.num_classes + 3] = is_seq

            # Node label for localization (1 if Trojan gate, 0 otherwise)
            node_labels[i] = 1 if attrs.get('is_trojan', False) else 0

        # Edge index
        edge_list = []
        for src, dst in G.edges():
            edge_list.append([node_to_idx[src], node_to_idx[dst]])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Graph label (1 if any node is Trojan, or if graph-level label is 1)
        graph_label = G.graph.get('label', 0)
        if node_labels.sum() > 0:
            graph_label = 1

        pyg_data = Data(
            x=torch.from_numpy(x),
            edge_index=edge_index,
            y=torch.tensor([graph_label], dtype=torch.long),
            node_y=torch.from_numpy(node_labels),
            num_nodes=num_nodes
        )
        pyg_data.node_names = nodes
        pyg_data.module_name = G.graph.get('module_name', 'circuit')

        return pyg_data
