"""
train.py: End-to-End Training and Evaluation Pipeline for Hardware Trojan GNN.

Trains the Dual-Task HardwareTrojanGNN model on Golden (Clean) vs. Trojan-Infected circuit benchmarks.
Evaluates both:
1. Macro Graph-Level Trojan Detection (Accuracy, Precision, Recall, F1, ROC-AUC)
2. Micro Gate-Level Trojan Localization (Gate Precision, Gate Recall, Gate F1)
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.data.dataset_loader import HardwareTrojanDataset
from src.models.gnn_detector import HardwareTrojanGNN
from src.evaluation.metrics import compute_detection_metrics, compute_localization_metrics


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_benchmark_dataset(
    num_samples_per_class: int = 35,
    max_nodes_per_graph: int = 5000
) -> List[Data]:
    """
    Collects a balanced set of Golden and Trojan-infected circuit netlists from local datasets.
    """
    dataset_handler = HardwareTrojanDataset(cache_dir="data_cache")
    cache_file = f"benchmark_train_{num_samples_per_class}per_class.pt"
    cached = dataset_handler.load_cached_graphs(cache_file)
    if cached:
        return cached

    print(f"Parsing fresh benchmark dataset ({num_samples_per_class} clean + {num_samples_per_class} trojan)...")
    graphs: List[Data] = []

    # 1. Clean / Golden Circuits
    clean_paths = []
    # From EPFL
    epfl_files = list(Path("datasets/epfl_benchmarks/arithmetic").glob("*.v")) + \
                 list(Path("datasets/epfl_benchmarks/random_control").glob("*.v"))
    clean_paths.extend([(str(p), 0) for p in epfl_files])

    # From TjFree
    tjfree_dir = Path("datasets/fast_accurate_ht_locations/data/TjFree")
    if tjfree_dir.exists():
        tjfree_files = list(tjfree_dir.glob("*/*.v"))[:50]
        clean_paths.extend([(str(p), 0) for p in tjfree_files])

    random.shuffle(clean_paths)
    clean_count = 0
    for path, label in clean_paths:
        if clean_count >= num_samples_per_class:
            break
        g = dataset_handler.load_single_verilog(path, label=0)
        if g and 10 < g.num_nodes <= max_nodes_per_graph:
            graphs.append(g)
            clean_count += 1

    # 2. Trojan-Infected Circuits
    tjin_dir = Path("datasets/fast_accurate_ht_locations/data/TjIn")
    trojan_paths = list(tjin_dir.glob("*/topModule.v"))
    random.shuffle(trojan_paths)

    trojan_count = 0
    for path in trojan_paths:
        if trojan_count >= num_samples_per_class:
            break
        g = dataset_handler.load_single_verilog(str(path), label=1)
        if g and 10 < g.num_nodes <= max_nodes_per_graph and (g.node_y == 1).sum() > 0:
            graphs.append(g)
            trojan_count += 1

    print(f"Collected {len(graphs)} total graphs ({clean_count} clean, {trojan_count} trojan-infected).")
    dataset_handler.save_graphs(graphs, cache_file)
    return graphs


def train_epoch(
    model: HardwareTrojanGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_graph: nn.Module,
    criterion_node: nn.Module,
    device: torch.device,
    lambda_node: float = 0.5
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)

        # Graph detection loss
        loss_graph = criterion_graph(out['graph_logits'], batch.y)

        # Node localization loss (only compute on graphs where node labels exist)
        loss_node = criterion_node(out['node_logits'], batch.node_y)

        # Combined multi-task loss
        loss = loss_graph + lambda_node * loss_node
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def evaluate(
    model: HardwareTrojanGNN,
    loader: DataLoader,
    device: torch.device
) -> Tuple[dict, dict]:
    model.eval()

    y_true_all = []
    y_pred_prob_all = []

    node_true_all = []
    node_pred_prob_all = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)

            # Graph probs
            graph_probs = torch.softmax(out['graph_logits'], dim=-1)[:, 1].cpu().numpy()
            y_true_all.extend(batch.y.cpu().numpy())
            y_pred_prob_all.extend(graph_probs)

            # Node probs
            node_probs = torch.softmax(out['node_logits'], dim=-1)[:, 1].cpu().numpy()
            node_true_all.extend(batch.node_y.cpu().numpy())
            node_pred_prob_all.extend(node_probs)

    det_metrics = compute_detection_metrics(np.array(y_true_all), np.array(y_pred_prob_all))
    loc_metrics = compute_localization_metrics(np.array(node_true_all), np.array(node_pred_prob_all))

    return det_metrics, loc_metrics


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print(f"  TRAINING HARDWARE TROJAN DETECTION & LOCALIZATION GNN ({device.type.upper()})")
    print("=" * 70)

    # 1. Dataset Collection
    graphs = collect_benchmark_dataset(num_samples_per_class=25, max_nodes_per_graph=6000)
    if len(graphs) < 10:
        print("Not enough graphs to train reliably.")
        return

    # Train / Test Split (80% / 20%)
    train_size = int(0.8 * len(graphs))
    test_size = len(graphs) - train_size
    train_graphs, test_graphs = random_split(graphs, [train_size, test_size])

    train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=4, shuffle=False)

    print(f"Train Dataset: {len(train_graphs)} circuits | Test Dataset: {len(test_graphs)} circuits")

    # 2. Model Initialization
    in_dim = graphs[0].x.size(1)
    model = HardwareTrojanGNN(
        in_channels=in_dim,
        hidden_channels=64,
        num_layers=3,
        backbone='gcn',
        dropout=0.15
    ).to(device)

    # Losses
    criterion_graph = nn.CrossEntropyLoss()
    # Heavily weight Trojan gate class (class 1) to handle 99.8% vs 0.2% class imbalance
    node_weights = torch.tensor([1.0, 30.0], device=device)
    criterion_node = nn.CrossEntropyLoss(weight=node_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # 3. Training Loop
    epochs = 20
    best_f1 = 0.0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best_ht_gnn.pt"

    print("\nStarting Training Epochs...")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion_graph, criterion_node, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            det_val, loc_val = evaluate(model, test_loader, device)
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {loss:.4f} | "
                  f"Det Acc: {det_val['accuracy']*100:.1f}% | "
                  f"Det F1: {det_val['f1']:.3f} | "
                  f"Det AUC: {det_val['roc_auc']:.3f} | "
                  f"Gate Rec: {loc_val['gate_recall']*100:.1f}%")

            if det_val['f1'] >= best_f1:
                best_f1 = det_val['f1']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'in_channels': in_dim,
                    'det_metrics': det_val,
                    'loc_metrics': loc_val
                }, str(best_checkpoint_path))

    # 4. Final Comprehensive Evaluation
    print("\n" + "=" * 70)
    print("                 FINAL EVALUATION BENCHMARK RESULTS")
    print("=" * 70)
    checkpoint = torch.load(str(best_checkpoint_path), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    final_det, final_loc = evaluate(model, test_loader, device)

    print(" [Task 1: Graph-Level Trojan Detection]")
    print(f"   - Accuracy  : {final_det['accuracy'] * 100:.2f}%")
    print(f"   - Precision : {final_det['precision'] * 100:.2f}%")
    print(f"   - Recall    : {final_det['recall'] * 100:.2f}%")
    print(f"   - F1-Score  : {final_det['f1']:.4f}")
    print(f"   - ROC-AUC   : {final_det['roc_auc']:.4f}")

    print("\n [Task 2: Node-Level Trojan Gate Localization]")
    print(f"   - Gate Recall (Trigger/Payload Coverage) : {final_loc['gate_recall'] * 100:.2f}%")
    print(f"   - Gate Precision                        : {final_loc['gate_precision'] * 100:.2f}%")
    print(f"   - Gate F1-Score                         : {final_loc['gate_f1']:.4f}")
    print(f"   - Total Trojan Gates in Test Set        : {final_loc['total_trojan_gates']}")
    print(f"   - Detected Trojan Gates                 : {final_loc['detected_trojan_gates']}")
    print("=" * 70)
    print(f"Best model weights saved to: {best_checkpoint_path.resolve()}\n")


if __name__ == "__main__":
    main()
