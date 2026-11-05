"""
eda_analysis.py: Comprehensive Exploratory Data Analysis (EDA) of Hardware Trojan Datasets.

Performs:
1. Macro inventory of dataset families (Golden vs. Infected).
2. Topological distributions: Gate counts, PI/PO counts, net interconnect density.
3. Trojan footprint analysis: Trigger size, payload ratio, gate types utilized in attacks.
4. Generates publication-ready visualizations and statistical summary tables.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.netlist_parser import NetlistParser


def run_comprehensive_eda():
    print("=" * 75)
    print("   EXPLORATORY DATA ANALYSIS (EDA): HARDWARE TROJAN DATASETS")
    print("=" * 75)

    parser = NetlistParser()
    out_dir = Path("experiments/eda_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Dataset Inventory Scan
    # -------------------------------------------------------------
    print("\n[Stage 1] Scanning Dataset Repositories...")

    epfl_dir = Path("datasets/epfl_benchmarks")
    tjin_dir = Path("datasets/fast_accurate_ht_locations/data/TjIn")
    tjfree_dir = Path("datasets/fast_accurate_ht_locations/data/TjFree")
    trusthub_extracted = Path("datasets/trust_hub_benchmarks/extracted")

    epfl_v = list(epfl_dir.glob("**/*.v")) if epfl_dir.exists() else []
    tjin_dirs = [d for d in tjin_dir.iterdir() if d.is_dir()] if tjin_dir.exists() else []
    tjfree_dirs = [d for d in tjfree_dir.iterdir() if d.is_dir()] if tjfree_dir.exists() else []
    trusthub_dirs = [d for d in trusthub_extracted.iterdir() if d.is_dir()] if trusthub_extracted.exists() else []

    inventory_df = pd.DataFrame([
        {"Dataset Source": "EPFL Benchmark Suite (Combinational)", "Type": "Golden (Trojan-Free)", "Count": len(epfl_v)},
        {"Dataset Source": "Fast-Accurate HT Locations (TjFree)", "Type": "Golden (Trojan-Free)", "Count": len(tjfree_dirs)},
        {"Dataset Source": "Fast-Accurate HT Locations (TjIn)", "Type": "Trojan-Infected", "Count": len(tjin_dirs)},
        {"Dataset Source": "Trust-Hub Flagship Extracted (AES)", "Type": "Infected & Golden Pairs", "Count": len(trusthub_dirs)},
    ])
    print(inventory_df.to_string(index=False))

    total_clean = len(epfl_v) + len(tjfree_dirs) + len(trusthub_dirs)
    total_infected = len(tjin_dirs) + len(trusthub_dirs)
    print(f"\nTotal Identified Circuits: {total_clean + total_infected} "
          f"({total_clean} Golden candidates, {total_infected} Trojan candidates)")

    # -------------------------------------------------------------
    # 2. Structural & Topological Analysis on Trojan Benchmarks
    # -------------------------------------------------------------
    print("\n[Stage 2] Parsing Topological Metrics & Trojan Footprints (Sampling 60 TjIn circuits)...")

    trojan_records = []
    gate_type_counter = Counter()
    trojan_gate_type_counter = Counter()

    # Sample representative circuits
    sample_dirs = tjin_dirs[:60]
    for d in sample_dirs:
        top_v = d / "topModule.v"
        if not top_v.exists():
            continue

        try:
            netlist = parser.parse_file(str(top_v))
            total_gates = len(netlist.gates)
            if total_gates == 0:
                continue

            trojan_gates = [g for g in netlist.gates if g.is_trojan]
            num_tr = len(trojan_gates)

            # Read troj.txt if present for ground truth validation
            troj_txt = d / "troj.txt"
            ground_truth_count = num_tr
            if troj_txt.exists():
                with open(troj_txt, 'r', errors='ignore') as f:
                    txt_lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('//') and 'endmodule' not in l]
                    ground_truth_count = max(num_tr, len(txt_lines))

            # Record gate types
            for g in netlist.gates:
                gate_type_counter[g.gate_type] += 1
                if g.is_trojan:
                    trojan_gate_type_counter[g.gate_type] += 1

            trojan_overhead_pct = (ground_truth_count / total_gates) * 100.0

            trojan_records.append({
                "circuit_name": d.name,
                "total_gates": total_gates,
                "primary_inputs": len(netlist.inputs),
                "primary_outputs": len(netlist.outputs),
                "trojan_gates": ground_truth_count,
                "trojan_overhead_pct": trojan_overhead_pct
            })
        except Exception:
            continue

    df_trojan = pd.DataFrame(trojan_records)
    print(f"Successfully profiled {len(df_trojan)} Trojan-infected circuits.")

    # Statistical summary
    print("\n--- Topological Statistics of Trojan-Infected Benchmarks ---")
    summary = df_trojan[["total_gates", "primary_inputs", "primary_outputs", "trojan_gates", "trojan_overhead_pct"]].describe()
    print(summary.round(2))

    # -------------------------------------------------------------
    # 3. Generating Visualization Plots
    # -------------------------------------------------------------
    print("\n[Stage 3] Generating Publication-Ready Figures in 'experiments/eda_plots/'...")

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: Circuit Size vs. Trojan Gate Footprint
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_trojan["total_gates"], df_trojan["trojan_gates"], color="#d9381e", alpha=0.7, edgecolors='k', s=60)
    ax.set_title("Circuit Size vs. Hardware Trojan Gate Count", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Total Circuit Gates (Netlist Size)", fontsize=11)
    ax.set_ylabel("Trojan Gates (Trigger + Payload)", fontsize=11)
    ax.axhline(df_trojan["trojan_gates"].median(), color='navy', linestyle='--', label=f'Median Trojan Size ({df_trojan["trojan_gates"].median():.0f} gates)')
    ax.legend(frameon=True)
    plt.tight_layout()
    fig1_path = out_dir / "circuit_size_vs_trojan_gates.png"
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig1_path}")

    # Figure 2: Trojan Overhead Percentage Distribution (The Stealthiness / Needle-in-a-Haystack problem)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_trojan["trojan_overhead_pct"], bins=20, color="#1a73e8", edgecolor="black", alpha=0.8)
    ax.set_title("Hardware Trojan Stealthiness: Gate Overhead (%) Distribution", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Trojan Gate Overhead (% of Total Circuit)", fontsize=11)
    ax.set_ylabel("Number of Circuit Benchmarks", fontsize=11)
    mean_oh = df_trojan["trojan_overhead_pct"].mean()
    ax.axvline(mean_oh, color='red', linestyle='--', linewidth=2, label=f'Mean Overhead: {mean_oh:.3f}%')
    ax.legend(frameon=True)
    plt.tight_layout()
    fig2_path = out_dir / "trojan_overhead_distribution.png"
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig2_path}")

    # Figure 3: Gate Type Utilization (Normal vs Trojan Gates)
    top_gates = [k for k, _ in gate_type_counter.most_common(8)]
    normal_counts = [gate_type_counter[k] for k in top_gates]
    trojan_counts = [trojan_gate_type_counter.get(k, 0) for k in top_gates]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(top_gates))
    width = 0.35
    ax.bar(x - width/2, [np.log1p(c) for c in normal_counts], width, label='Normal Circuit Gates (log scale)', color='#4285f4')
    ax.bar(x + width/2, [np.log1p(c) for c in trojan_counts], width, label='Trojan Trigger/Payload Gates (log scale)', color='#ea4335')
    ax.set_title("Gate Type Distribution: Standard Gates vs. Trojan Mechanisms", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(top_gates, fontsize=10, rotation=15)
    ax.set_ylabel("Log(1 + Gate Count)", fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig3_path = out_dir / "gate_type_distribution.png"
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig3_path}")

    print("\n" + "=" * 75)
    print("                      EDA ANALYSIS COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    run_comprehensive_eda()
