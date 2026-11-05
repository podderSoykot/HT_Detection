"""
anomaly_detector.py: Unseen / Zero-Day Hardware Trojan Anomaly Detection.

Learns the distribution of Golden (Trojan-Free) circuit representations
and flags out-of-distribution / unfamiliar netlists as potential zero-day Trojans.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class EmbeddingAutoencoder(nn.Module):
    """Reconstruction Autoencoder for circuit graph latent embeddings."""

    def __init__(self, embed_dim: int = 192, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Linear(96, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Linear(96, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates MSE reconstruction error per circuit embedding."""
        self.eval()
        with torch.no_grad():
            x_recon, _ = self.forward(x)
            mse = torch.mean((x - x_recon) ** 2, dim=-1)
        return mse


class UnseenTrojanDetector:
    """
    Detector for Unknown / Unseen Hardware Trojans.
    Combines:
    1. Deep Autoencoder reconstruction anomaly score
    2. Isolation Forest density score on GNN graph embeddings
    """

    def __init__(self, embed_dim: int = 192, latent_dim: int = 32, contamination: float = 0.05):
        self.embed_dim = embed_dim
        self.autoencoder = EmbeddingAutoencoder(embed_dim, latent_dim)
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42
        )
        self.threshold: float = 0.0
        self.is_fitted: bool = False

    def fit_golden_manifold(
        self,
        golden_embeddings: torch.Tensor,
        epochs: int = 40,
        lr: float = 1e-3,
        percentile_thresh: float = 95.0
    ):
        """
        Fits the anomaly model on embeddings of verified Golden / Clean circuits only.
        """
        self.autoencoder.train()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        # Train Autoencoder
        dataset = torch.utils.data.TensorDataset(golden_embeddings)
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(16, len(golden_embeddings)), shuffle=True)

        for _ in range(epochs):
            for (batch_x,) in loader:
                if batch_x.size(0) <= 1:
                    continue
                optimizer.zero_grad()
                recon, _ = self.autoencoder(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

        # Compute threshold from Golden reconstruction errors
        errors = self.autoencoder.compute_reconstruction_error(golden_embeddings).cpu().numpy()
        self.threshold = float(np.percentile(errors, percentile_thresh))

        # Fit Isolation Forest
        np_embeddings = golden_embeddings.detach().cpu().numpy()
        self.iso_forest.fit(np_embeddings)
        self.is_fitted = True

    def score_anomaly(self, circuit_embeddings: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Scores incoming third-party IP circuits for zero-day/unseen Trojan suspicion.
        Returns:
            - 'recon_error': Continuous reconstruction error
            - 'anomaly_score': Normalized [0, 1] anomaly score
            - 'is_unseen_trojan': Boolean flag (True if abnormal)
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector must be fitted on Golden embeddings first.")

        errors = self.autoencoder.compute_reconstruction_error(circuit_embeddings).cpu().numpy()
        np_embeddings = circuit_embeddings.detach().cpu().numpy()

        # Isolation forest decision function (lower means more anomalous)
        if_scores = -self.iso_forest.score_samples(np_embeddings)

        # Normalized anomaly score combining reconstruction and isolation forest
        norm_recon = np.clip(errors / (self.threshold + 1e-6), 0.0, 3.0) / 3.0
        norm_if = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-6)
        combined_score = 0.6 * norm_recon + 0.4 * norm_if

        is_anomalous = errors > self.threshold

        return {
            'recon_error': errors,
            'anomaly_score': np.clip(combined_score, 0.0, 1.0),
            'is_unseen_trojan': is_anomalous
        }
