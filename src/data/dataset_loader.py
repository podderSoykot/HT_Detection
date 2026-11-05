"""
dataset_loader.py: High-throughput benchmark dataset loader for Hardware Trojan Detection.

Crawls datasets across:
- EPFL Benchmarks (Combinational Golden reference designs)
- Trust-Hub Benchmarks (Extracted and TjIn / TjFree circuits)
- Fast-Accurate HT Location datasets
Produces PyTorch Geometric graph datasets ready for GNN training.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from src.data.netlist_parser import NetlistParser
from src.data.graph_builder import CircuitGraphBuilder


class HardwareTrojanDataset:
    """Manages collection, parsing, and caching of circuit graph datasets."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.parser = NetlistParser()
        self.builder = CircuitGraphBuilder()
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_epfl_benchmarks(self, root_dir: str = "datasets/epfl_benchmarks") -> List[Data]:
        """Loads EPFL combinational benchmarks as Golden (Trojan-Free, label=0) graphs."""
        root = Path(root_dir)
        graphs: List[Data] = []
        v_files = sorted(list(root.glob("**/*.v")))

        for v_file in v_files:
            try:
                netlist = self.parser.parse_file(str(v_file))
                if not netlist.gates:
                    continue
                nx_g = self.builder.build_networkx(netlist, graph_label=0)
                data = self.builder.to_pyg_data(nx_g)
                data.circuit_family = "EPFL"
                data.is_trojan = False
                graphs.append(data)
            except Exception as e:
                print(f"Warning: Failed to parse {v_file.name}: {e}")

        return graphs

    def load_single_verilog(self, file_path: str, label: int = 0) -> Optional[Data]:
        """Parses a single Verilog file directly into a PyG Data object."""
        try:
            netlist = self.parser.parse_file(file_path)
            nx_g = self.builder.build_networkx(netlist, graph_label=label)
            data = self.builder.to_pyg_data(nx_g)
            data.is_trojan = bool(label)
            return data
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None

    def save_graphs(self, graphs: List[Data], filename: str):
        """Caches graph dataset to disk."""
        target_path = self.cache_dir / filename
        torch.save(graphs, str(target_path))
        print(f"Saved {len(graphs)} circuit graphs to {target_path}")

    def load_cached_graphs(self, filename: str) -> Optional[List[Data]]:
        """Loads cached graph dataset from disk."""
        target_path = self.cache_dir / filename
        if target_path.exists():
            graphs = torch.load(str(target_path), weights_only=False)
            print(f"Loaded {len(graphs)} graphs from cache ({target_path})")
            return graphs
        return None
