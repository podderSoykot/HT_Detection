"""
visualize_circuit_graph.py: Visualizes the parsed Circuit Graph and Hardware Trojan Subgraph.

Generates:
1. Trojan Subgraph View: Isolates the malicious trigger tree, payload, and host interface nodes.
2. Full Circuit Graph Macro View: Displays the overarching network topology.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.data.netlist_parser import NetlistParser
from src.data.graph_builder import CircuitGraphBuilder


def visualize_trojan_circuit():
    print("=== Generating Circuit Graph Visualizations ===")
    parser = NetlistParser()
    builder = CircuitGraphBuilder()
    out_dir = Path("experiments/eda_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse Trojan-infected benchmark: c5315_T047
    netlist_path = "datasets/fast_accurate_ht_locations/data/TjIn/c5315_T047/topModule.v"
    print(f"Loading netlist: {netlist_path}")
    netlist = parser.parse_file(netlist_path)
    G = builder.build_networkx(netlist, graph_label=1)

    print(f"Full Graph Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # Find Trojan nodes
    trojan_nodes = [n for n, d in G.nodes(data=True) if d.get('is_trojan', False)]
    print(f"Identified Trojan Nodes ({len(trojan_nodes)}): {trojan_nodes}")

    # -------------------------------------------------------------
    # Visual 1: Trojan Neighborhood Subgraph (Trigger + Payload + 1-hop Boundary)
    # -------------------------------------------------------------
    # Get 1-hop predecessors and successors to see how the Trojan taps into host signals
    subgraph_nodes = set(trojan_nodes)
    for n in trojan_nodes:
        subgraph_nodes.update(G.predecessors(n))
        subgraph_nodes.update(G.successors(n))

    H = G.subgraph(subgraph_nodes).copy()
    print(f"Extracted Trojan Neighborhood Subgraph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Layout using Kamada-Kawai or spring
    try:
        pos = nx.kamada_kawai_layout(H)
    except Exception:
        pos = nx.spring_layout(H, seed=42, k=0.8)

    # Separate node categories
    tr_nodes = [n for n in H.nodes() if H.nodes[n].get('is_trojan', False)]
    host_gates = [n for n in H.nodes() if not H.nodes[n].get('is_trojan', False) and H.nodes[n].get('gate_type') not in ('PI', 'PO')]
    pi_po_nodes = [n for n in H.nodes() if H.nodes[n].get('gate_type') in ('PI', 'PO')]

    # Draw host gates (Golden / Clean)
    nx.draw_networkx_nodes(
        H, pos, nodelist=host_gates,
        node_color='#93c5fd', node_size=750,
        edgecolors='#1d4ed8', linewidths=2.0,
        label='Clean Host Gates', ax=ax
    )

    # Draw Host Ports (PI/PO)
    if pi_po_nodes:
        nx.draw_networkx_nodes(
            H, pos, nodelist=pi_po_nodes,
            node_color='#bbf7d0', node_size=850, node_shape='s',
            edgecolors='#15803d', linewidths=2.0,
            label='Primary I/O Ports', ax=ax
        )

    # Draw Trojan Gates (Malicious Trigger & Payload)
    nx.draw_networkx_nodes(
        H, pos, nodelist=tr_nodes,
        node_color='#ef4444', node_size=1100,
        edgecolors='#7f1d1d', linewidths=3.0,
        label='Hardware Trojan Gates (Trigger/Payload)', ax=ax
    )

    # Draw Edges
    nx.draw_networkx_edges(
        H, pos, edge_color='#64748b',
        arrows=True, arrowsize=18, arrowstyle='-|>',
        width=1.6, alpha=0.8, ax=ax
    )

    # Labels: Node Name + Gate Type
    labels = {}
    for n in H.nodes():
        gtype = H.nodes[n].get('gate_type', '')
        short_name = n.replace("assign_gate_", "g_")
        labels[n] = f"{short_name}\n[{gtype}]"

    nx.draw_networkx_labels(H, pos, labels=labels, font_size=8, font_weight='bold', ax=ax)

    ax.set_title(
        f"Circuit Graph: Hardware Trojan Subgraph in Netlist ({netlist.module_name})\n"
        f"Red: Trojan Attack Nodes | Blue: Clean Host Circuit Interface",
        fontsize=14, fontweight='bold', pad=15
    )
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    ax.axis('off')

    plt.tight_layout()
    subgraph_png = out_dir / "trojan_subgraph_visualization.png"
    fig.savefig(subgraph_png, dpi=300)
    plt.close(fig)
    print(f"Saved: {subgraph_png}")

    # -------------------------------------------------------------
    # Visual 2: Macro Topology of a Standard Circuit
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    sample_sub = nx.ego_graph(G, trojan_nodes[0], radius=2)
    pos2 = nx.spring_layout(sample_sub, seed=42)

    colors = ['#ef4444' if sample_sub.nodes[n].get('is_trojan', False) else '#60a5fa' for n in sample_sub.nodes()]
    nx.draw_networkx_nodes(sample_sub, pos2, node_color=colors, node_size=350, edgecolors='black', ax=ax)
    nx.draw_networkx_edges(sample_sub, pos2, edge_color='#94a3b8', alpha=0.6, arrows=True, ax=ax)

    ax.set_title(f"Macro Graph Topology (2-Hop Ego Graph around Trojan Trigger)", fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    macro_png = out_dir / "circuit_graph_macro_view.png"
    fig.savefig(macro_png, dpi=300)
    plt.close(fig)
    print(f"Saved: {macro_png}")

    print("\nVisualizations successfully generated!")


if __name__ == "__main__":
    visualize_trojan_circuit()
