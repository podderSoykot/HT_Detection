"""
hierarchical_graph_plot.py: Publication-quality hierarchical circuit schematic visualizer.

Constructs an ultra-clean Left-to-Right schematic graph layout:
- Inputs on the left
- Trigger comparator layers in the middle
- Trigger enabler and Payload on the right
- No overlapping labels, smooth spline arrows, colored functional subcircuit groupings.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np

from src.data.netlist_parser import NetlistParser
from src.data.graph_builder import CircuitGraphBuilder


def generate_hierarchical_schematic():
    print("Generating ultra-clean hierarchical circuit graph schematic...")
    parser = NetlistParser()
    builder = CircuitGraphBuilder()
    out_dir = Path("experiments/eda_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    netlist_path = "datasets/fast_accurate_ht_locations/data/TjIn/c5315_T047/topModule.v"
    netlist = parser.parse_file(netlist_path)
    G = builder.build_networkx(netlist, graph_label=1)

    trojan_nodes = [n for n, d in G.nodes(data=True) if d.get('is_trojan', False)]

    # Collect Trojan neighborhood
    subgraph_nodes = set(trojan_nodes)
    for n in trojan_nodes:
        subgraph_nodes.update(G.predecessors(n))
        subgraph_nodes.update(G.successors(n))

    H = G.subgraph(subgraph_nodes).copy()

    # Define exact hierarchical columns for logic schematic flow
    # Layer 0: Host Driving Signals (Inputs into the Trojan)
    # Layer 1: First-stage Trigger Gates (troj470U2, troj470U3, troj470U4, troj470U6, troj470U7)
    # Layer 2: Second-stage Trigger Gate (troj470U5)
    # Layer 3: Final Trigger Decision Gate (troj470U1 - Triggeren470)
    # Layer 4: Original Victim Net (tempN8090) and Payload Gate (trojan470)
    # Layer 5: Corrupted Circuit Net (N8090 / Sinks)

    layer_map = {}

    # Explicit Trojan Gate Layers
    layer_map['troj470U7'] = 1  # BUF
    layer_map['troj470U6'] = 1  # NOR
    layer_map['troj470U4'] = 1  # NAND
    layer_map['troj470U3'] = 1  # NAND
    layer_map['troj470U2'] = 1  # NAND
    layer_map['troj470U5'] = 2  # NAND (combines U6, U7, N1381)
    layer_map['troj470U1'] = 3  # NOR (combines U2, U3, U4, U5)
    layer_map['trojan470'] = 4  # XOR Payload

    for n in H.nodes():
        if n in layer_map:
            continue
        # Predecessors of Layer 1 or 2 are Layer 0 (Host Signals)
        succs = list(H.successors(n))
        if 'trojan470' in succs:
            layer_map[n] = 3.5  # Victim wire (tempN8090)
        elif any(layer_map.get(s, 0) == 1 for s in succs):
            layer_map[n] = 0
        elif any(layer_map.get(s, 0) == 2 for s in succs):
            layer_map[n] = 0.5
        elif any(layer_map.get(s, 0) == 4 for s in succs):
            layer_map[n] = 3.5
        else:
            preds = list(H.predecessors(n))
            if any(layer_map.get(p, 0) == 4 for p in preds):
                layer_map[n] = 5  # Corrupted output wire/sinks
            else:
                layer_map[n] = 0

    # Group nodes by layer
    layers = defaultdict(list)
    for n, l in layer_map.items():
        layers[l].append(n)

    # Compute (x, y) coordinates
    pos = {}
    sorted_levels = sorted(layers.keys())

    # Map level keys to x-coords
    x_positions = {0: 1.0, 0.5: 2.5, 1: 4.5, 2: 7.5, 3: 10.5, 3.5: 11.8, 4: 14.0, 5: 16.5}

    for lvl in sorted_levels:
        nodes_in_lvl = sorted(layers[lvl])
        x = x_positions.get(lvl, lvl * 3.0)
        num_nodes = len(nodes_in_lvl)
        y_span = max(8.0, num_nodes * 1.3)
        y_vals = np.linspace(y_span / 2, -y_span / 2, num_nodes) if num_nodes > 1 else [0.0]
        for node, y in zip(nodes_in_lvl, y_vals):
            pos[node] = (x, y)

    # -------------------------------------------------------------
    # Render with Matplotlib
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(18, 10), facecolor='#ffffff')
    ax.set_facecolor('#f8fafc')

    # Draw Subcircuit Boundary Bounding Boxes
    # 1. Clean Host Inputs Box
    rect_host = patches.FancyBboxPatch(
        (0.2, -6.5), 2.8, 13.0,
        boxstyle="round,pad=0.3",
        linewidth=1.5, edgecolor="#93c5fd", facecolor="#eff6ff",
        linestyle="--", alpha=0.6, zorder=1
    )
    ax.add_patch(rect_host)
    ax.text(1.6, 6.8, "HOST CIRCUIT INPUT SIGNALS\n(Tapped by Trojan)",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#1e40af")

    # 2. Trojan Trigger Tree Box
    rect_trig = patches.FancyBboxPatch(
        (3.5, -6.0), 7.8, 12.0,
        boxstyle="round,pad=0.4",
        linewidth=2.0, edgecolor="#fca5a5", facecolor="#fef2f2",
        linestyle="--", alpha=0.6, zorder=1
    )
    ax.add_patch(rect_trig)
    ax.text(7.4, 6.3, "MALICIOUS TROJAN TRIGGER COMPARATOR TREE\n(Activates when rare condition occurs)",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#991b1b")

    # 3. Payload Box
    rect_pl = patches.FancyBboxPatch(
        (12.5, -3.5), 4.8, 7.0,
        boxstyle="round,pad=0.4",
        linewidth=2.0, edgecolor="#fde047", facecolor="#fefce8",
        linestyle="--", alpha=0.6, zorder=1
    )
    ax.add_patch(rect_pl)
    ax.text(14.9, 3.8, "TROJAN PAYLOAD INJECTION\n(Alters functional output)",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#854d0e")

    # Draw Edges with curved spline routing
    for u, v in H.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        is_trojan_edge = H.nodes[u].get('is_trojan', False) and H.nodes[v].get('is_trojan', False)
        edge_color = '#dc2626' if is_trojan_edge else '#64748b'
        width = 2.0 if is_trojan_edge else 1.3

        rad = 0.08 if (y2 - y1) >= 0 else -0.08
        ax.annotate(
            "",
            xy=(x2 - 0.45, y2), xycoords='data',
            xytext=(x1 + 0.45, y1), textcoords='data',
            arrowprops=dict(
                arrowstyle="-|>",
                color=edge_color,
                lw=width,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=2, shrinkB=2,
                mutation_scale=14
            ),
            zorder=2
        )

    # Draw Nodes as clean rectangular cards
    for n in H.nodes():
        x, y = pos[n]
        is_tr = H.nodes[n].get('is_trojan', False)
        gtype = H.nodes[n].get('gate_type', 'NET')

        if is_tr:
            if n == 'trojan470':
                card_bg = '#fef08a'
                border_color = '#ca8a04'
                badge_bg = '#ca8a04'
            else:
                card_bg = '#fee2e2'
                border_color = '#dc2626'
                badge_bg = '#dc2626'
            badge_fg = '#ffffff'
        else:
            card_bg = '#ffffff'
            border_color = '#3b82f6'
            badge_bg = '#3b82f6'
            badge_fg = '#ffffff'

        # Main node card
        card = patches.FancyBboxPatch(
            (x - 0.45, y - 0.35), 0.9, 0.7,
            boxstyle="round,pad=0.1",
            linewidth=2.0, edgecolor=border_color, facecolor=card_bg,
            zorder=3
        )
        ax.add_patch(card)

        # Gate type badge
        ax.text(x, y + 0.12, gtype, ha='center', va='center',
                fontsize=8.5, fontweight='black', color=badge_bg, zorder=4)

        # Instance Name
        short_name = n.replace("assign_gate_", "g")
        ax.text(x, y - 0.14, short_name, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='#1e293b', zorder=4)

    ax.set_xlim(-0.5, 18.0)
    ax.set_ylim(-7.5, 7.8)
    ax.axis('off')

    plt.title("Hardware Trojan Gate-Level Circuit Schematic (c5315_T047)\nHierarchical Left-to-Right Signal Flow",
              fontsize=16, fontweight='bold', pad=18, color='#0f172a')

    plt.tight_layout()
    schematic_path = out_dir / "hierarchical_schematic.png"
    fig.savefig(schematic_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Hierarchical Schematic saved to: {schematic_path}")


if __name__ == "__main__":
    generate_hierarchical_schematic()
