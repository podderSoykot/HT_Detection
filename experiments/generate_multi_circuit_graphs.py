"""
generate_multi_circuit_graphs.py: Builds and visualizes separate circuit graphs across distinct benchmark families.

Extracts and visualizes:
1. EPFL Adder (Golden Arithmetic Circuit)
2. ISCAS c3540_T018 (Trojan-Infected ALU)
3. ISCAS c5315_T047 (Trojan-Infected 9-bit ALU)
4. ISCAS c6288_T067 (Trojan-Infected 16x16 Multiplier)
5. Microcontroller PIC16F84-T100 (Trojan-Infected Embedded Processor)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import torch

from src.data.netlist_parser import NetlistParser
from src.data.graph_builder import CircuitGraphBuilder


def build_and_render_all():
    print("=" * 75)
    print("   BUILDING INDIVIDUAL CIRCUIT GRAPHS ACROSS MULTIPLE FAMILIES")
    print("=" * 75)

    parser = NetlistParser()
    builder = CircuitGraphBuilder()
    out_dir = Path("experiments/eda_plots/circuit_graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    circuit_specs = [
        {
            "id": "adder_golden",
            "title": "EPFL Adder (Golden 128-bit Arithmetic)",
            "path": "datasets/epfl_benchmarks/arithmetic/adder.v",
            "type": "Golden",
            "label": 0
        },
        {
            "id": "c3540_t018",
            "title": "ISCAS-85 c3540_T018 (Trojan 8-bit ALU)",
            "path": "datasets/fast_accurate_ht_locations/data/TjIn/c3540_T018/topModule.v",
            "type": "Trojan-Infected",
            "label": 1
        },
        {
            "id": "c5315_t047",
            "title": "ISCAS-85 c5315_T047 (Trojan 9-bit ALU/Selector)",
            "path": "datasets/fast_accurate_ht_locations/data/TjIn/c5315_T047/topModule.v",
            "type": "Trojan-Infected",
            "label": 1
        },
        {
            "id": "c6288_t067",
            "title": "ISCAS-85 c6288_T067 (Trojan 16x16 Multiplier)",
            "path": "datasets/fast_accurate_ht_locations/data/TjIn/c6288_T067/topModule.v",
            "type": "Trojan-Infected",
            "label": 1
        },
        {
            "id": "pic16f84_t100",
            "title": "PIC16F84-T100 (Trojan Microcontroller Core)",
            "path": "datasets/fast_accurate_ht_locations/data/TjIn/PIC16F84-T100/topModule.v",
            "type": "Trojan-Infected",
            "label": 1
        }
    ]

    graph_objects = []

    # 1. Build individual PyG and NetworkX graphs
    for spec in circuit_specs:
        print(f"\nProcessing Circuit Graph: {spec['title']}...")
        netlist = parser.parse_file(spec['path'])
        nx_g = builder.build_networkx(netlist, graph_label=spec['label'])
        pyg_data = builder.to_pyg_data(nx_g)

        tr_count = int((pyg_data.node_y == 1).sum())
        spec['num_nodes'] = nx_g.number_of_nodes()
        spec['num_edges'] = nx_g.number_of_edges()
        spec['trojan_gates'] = tr_count
        spec['nx_graph'] = nx_g
        spec['pyg_data'] = pyg_data

        print(f"  -> Built Graph: {spec['num_nodes']} Nodes, {spec['num_edges']} Edges, {tr_count} Trojan Gates")
        graph_objects.append(spec)

        # Render Individual Circuit Visualization
        render_single_circuit(spec, out_dir / f"{spec['id']}.png")

    # 2. Render Comparative 4-Panel Overview Gallery
    render_comparative_gallery(graph_objects[:4], out_dir / "multi_circuit_comparison_gallery.png")

    print("\n" + "=" * 75)
    print(f"All individual circuit graphs built and saved to: {out_dir.resolve()}")
    print("=" * 75)


def render_single_circuit(spec: dict, save_path: Path):
    """Renders a clean topological view for a single circuit."""
    G = spec['nx_graph']
    is_trojan = spec['label'] == 1

    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#ffffff')
    ax.set_facecolor('#f8fafc')

    # Subsample if graph is massive (> 1200 nodes) for crystal-clear visual rendering
    if G.number_of_nodes() > 600:
        if is_trojan:
            tr_nodes = [n for n, d in G.nodes(data=True) if d.get('is_trojan', False)]
            seed_node = tr_nodes[0] if tr_nodes else list(G.nodes())[0]
            sub = nx.ego_graph(G, seed_node, radius=2)
            if sub.number_of_nodes() < 40 and len(tr_nodes) > 1:
                sub = nx.ego_graph(G, seed_node, radius=3)
            title_suffix = f"(Targeted Ego-Subgraph: {sub.number_of_nodes()} Nodes around Insertion Point)"
        else:
            sample_nodes = list(G.nodes())[:350]
            sub = G.subgraph(sample_nodes).copy()
            title_suffix = f"(Core Subgraph: {sub.number_of_nodes()} Gates & Nets)"
    else:
        sub = G
        title_suffix = f"(Complete Circuit Graph: {sub.number_of_nodes()} Nodes)"

    try:
        pos = nx.kamada_kawai_layout(sub)
    except Exception:
        pos = nx.spring_layout(sub, seed=42, k=0.4)

    # Node coloring
    clean_nodes = [n for n in sub.nodes() if not sub.nodes[n].get('is_trojan', False) and sub.nodes[n].get('gate_type') not in ('PI', 'PO')]
    pi_po_nodes = [n for n in sub.nodes() if sub.nodes[n].get('gate_type') in ('PI', 'PO')]
    tr_nodes = [n for n in sub.nodes() if sub.nodes[n].get('is_trojan', False)]

    # Draw host gates
    nx.draw_networkx_nodes(sub, pos, nodelist=clean_nodes, node_color='#60a5fa', node_size=180, edgecolors='#1e40af', alpha=0.9, label='Clean Logic Gate', ax=ax)
    # Draw PI/PO
    if pi_po_nodes:
        nx.draw_networkx_nodes(sub, pos, nodelist=pi_po_nodes, node_color='#34d399', node_size=240, node_shape='s', edgecolors='#065f46', alpha=0.9, label='Primary Port (PI/PO)', ax=ax)
    # Draw Trojan gates
    if tr_nodes:
        nx.draw_networkx_nodes(sub, pos, nodelist=tr_nodes, node_color='#ef4444', node_size=450, node_shape='o', edgecolors='#991b1b', linewidths=2.5, label=f'Trojan Gates ({len(tr_nodes)} in view)', ax=ax)

    # Draw edges
    edge_colors = ['#dc2626' if sub.nodes[u].get('is_trojan', False) and sub.nodes[v].get('is_trojan', False) else '#94a3b8' for u, v in sub.edges()]
    widths = [2.0 if sub.nodes[u].get('is_trojan', False) and sub.nodes[v].get('is_trojan', False) else 0.8 for u, v in sub.edges()]
    nx.draw_networkx_edges(sub, pos, edge_color=edge_colors, width=widths, alpha=0.7, arrows=True, arrowsize=10, ax=ax)

    status_tag = f"TROJAN DETECTED: {spec['trojan_gates']} Malicious Gates" if is_trojan else "VERIFIED GOLDEN: 0 Malicious Gates"
    tag_color = "#dc2626" if is_trojan else "#16a34a"

    ax.set_title(f"Circuit Graph: {spec['title']}\n{title_suffix}", fontsize=13, fontweight='bold', pad=12)
    ax.text(0.02, 0.96, f"[{status_tag}] | Total Netlist Size: {spec['num_nodes']} Nodes, {spec['num_edges']} Edges",
            transform=ax.transAxes, fontsize=10, fontweight='bold', color=tag_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor=tag_color, alpha=0.9))

    ax.legend(loc='lower right', frameon=True, fontsize=9)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved individual plot: {save_path.name}")


def render_comparative_gallery(specs: list, save_path: Path):
    """Renders a 4-panel side-by-side comparison of different circuit graph topologies."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor='#ffffff')
    axes = axes.flatten()

    for idx, (spec, ax) in enumerate(zip(specs, axes)):
        ax.set_facecolor('#f8fafc')
        G = spec['nx_graph']
        is_trojan = spec['label'] == 1

        if is_trojan:
            tr_nodes = [n for n, d in G.nodes(data=True) if d.get('is_trojan', False)]
            seed_node = tr_nodes[0] if tr_nodes else list(G.nodes())[0]
            sub = nx.ego_graph(G, seed_node, radius=2)
            subtitle = f"Trojan-Infected | {spec['num_nodes']} Total Nodes | {spec['trojan_gates']} Trojan Gates"
            title_color = "#dc2626"
        else:
            sample_nodes = list(G.nodes())[:250]
            sub = G.subgraph(sample_nodes).copy()
            subtitle = f"Golden Reference | {spec['num_nodes']} Total Nodes | 0 Trojans"
            title_color = "#16a34a"

        try:
            pos = nx.kamada_kawai_layout(sub)
        except Exception:
            pos = nx.spring_layout(sub, seed=42, k=0.5)

        clean_nodes = [n for n in sub.nodes() if not sub.nodes[n].get('is_trojan', False)]
        tr_nodes = [n for n in sub.nodes() if sub.nodes[n].get('is_trojan', False)]

        nx.draw_networkx_nodes(sub, pos, nodelist=clean_nodes, node_color='#60a5fa', node_size=120, edgecolors='#1e40af', alpha=0.85, ax=ax)
        if tr_nodes:
            nx.draw_networkx_nodes(sub, pos, nodelist=tr_nodes, node_color='#ef4444', node_size=350, edgecolors='#991b1b', linewidths=2.0, ax=ax)

        nx.draw_networkx_edges(sub, pos, edge_color='#94a3b8', alpha=0.6, width=0.8, arrows=True, arrowsize=8, ax=ax)

        ax.set_title(f"Graph {idx+1}: {spec['title']}\n{subtitle}", fontsize=12, fontweight='bold', color=title_color, pad=10)
        ax.axis('off')

    plt.suptitle("Comparative Multi-Circuit Graph Topologies Across Benchmark Families", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nComparative 4-Panel Gallery saved to: {save_path.name}")


if __name__ == "__main__":
    build_and_render_all()
