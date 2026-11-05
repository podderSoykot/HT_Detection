"""
test_parser.py: Verifies netlist parsing and circuit graph construction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_loader import HardwareTrojanDataset


def main():
    print("=== Testing Hardware Trojan Netlist-to-Graph Pipeline ===")
    loader = HardwareTrojanDataset()

    # Test EPFL benchmarks
    print("\n[1/2] Loading EPFL Combinational Benchmarks...")
    epfl_graphs = loader.load_epfl_benchmarks("datasets/epfl_benchmarks")
    print(f"Successfully converted {len(epfl_graphs)} EPFL circuits to PyG graphs.")

    for g in epfl_graphs[:5]:
        print(f" - Circuit: {g.module_name:<12} | Nodes: {g.num_nodes:<6} | Edges: {g.edge_index.shape[1]:<6} | Label: {g.y.item()}")

    # Test single file
    print("\n[2/2] Parsing single circuit test...")
    sample_path = "datasets/epfl_benchmarks/arithmetic/adder.v"
    sample_g = loader.load_single_verilog(sample_path, label=0)
    if sample_g:
        print(f"Sample Graph: {sample_g.module_name} | Node feature dim: {sample_g.x.shape[1]}")
        print("Feature Vector Sample (first node):")
        print(sample_g.x[0].tolist())

    print("\n=== Verification Completed Successfully ===")


if __name__ == "__main__":
    main()
