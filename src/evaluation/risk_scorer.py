"""
risk_scorer.py: Evaluates IP Trustworthiness and Computes Quantitative Security Risk Score.

Synthesizes:
1. Known Trojan Detection Confidence (from GNN)
2. Unknown / Unseen Anomaly Score (from Anomaly Detector)
3. Localization Severity & Suspicious Gate Footprint
Produces an explainable Security Audit Report with CVSS-like Risk Score [0.0 - 10.0].
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class SecurityAuditResult:
    module_name: str
    security_risk_score: float         # 0.0 to 10.0
    risk_level: str                    # LOW, MEDIUM, HIGH, CRITICAL
    trustworthiness_pct: float         # 0.0% to 100.0%
    known_trojan_prob: float           # 0.0 to 1.0
    unseen_anomaly_score: float        # 0.0 to 1.0
    suspicious_gate_count: int
    suspicious_gate_ratio: float       # Percentage of total gates
    top_suspicious_gates: List[str]
    verdict: str


class IPTrustworthinessAssessor:
    """Computes final IP trustworthiness metrics and security risk score."""

    def __init__(
        self,
        weight_known: float = 0.50,
        weight_unseen: float = 0.30,
        weight_localization: float = 0.20
    ):
        self.w_known = weight_known
        self.w_unseen = weight_unseen
        self.w_loc = weight_localization

    def assess(
        self,
        module_name: str,
        known_trojan_prob: float,
        unseen_anomaly_score: float,
        gate_probabilities: np.ndarray,
        gate_names: List[str],
        gate_threshold: float = 0.5
    ) -> SecurityAuditResult:
        """
        Assesses a third-party IP and computes the Security Risk Score.
        """
        # Find suspicious gates
        suspicious_indices = np.where(gate_probabilities >= gate_threshold)[0]
        suspicious_count = len(suspicious_indices)
        total_gates = max(1, len(gate_names))
        suspicious_ratio = suspicious_count / total_gates

        # Sort top suspicious gates by probability
        ranked_indices = np.argsort(gate_probabilities)[::-1]
        top_gates = [
            f"{gate_names[idx]} (p={gate_probabilities[idx]:.3f})"
            for idx in ranked_indices[:min(10, suspicious_count)]
        ]

        # Multi-factor Risk Index in [0.0, 1.0]
        # Trojan presence is high if either known detector triggers or anomaly is severe
        detection_composite = max(
            known_trojan_prob,
            0.7 * known_trojan_prob + 0.3 * unseen_anomaly_score
        )
        footprint_factor = min(1.0, np.sqrt(suspicious_ratio * 20.0))  # sensitive even to small payloads

        raw_risk = (
            self.w_known * known_trojan_prob +
            self.w_unseen * unseen_anomaly_score +
            self.w_loc * footprint_factor
        )

        # Scale to 0.0 - 10.0
        risk_score = round(float(np.clip(raw_risk * 10.0, 0.0, 10.0)), 2)

        # Trustworthiness is inverse of risk
        trust_pct = round(max(0.0, 100.0 - (risk_score * 10.0)), 1)

        # Categorize risk level
        if risk_score >= 8.5:
            level = "CRITICAL"
            verdict = "REJECT: Confirmed Hardware Trojan detected with high payload severity."
        elif risk_score >= 6.0:
            level = "HIGH"
            verdict = "WARNING: Highly suspicious logic structures identified. Quarantine IP."
        elif risk_score >= 3.0:
            level = "MEDIUM"
            verdict = "CAUTION: Moderate structural anomalies detected. Manual review recommended."
        else:
            level = "LOW"
            verdict = "APPROVED: Passed trustworthiness evaluation. Cleared for integration."

        return SecurityAuditResult(
            module_name=module_name,
            security_risk_score=risk_score,
            risk_level=level,
            trustworthiness_pct=trust_pct,
            known_trojan_prob=round(float(known_trojan_prob), 4),
            unseen_anomaly_score=round(float(unseen_anomaly_score), 4),
            suspicious_gate_count=suspicious_count,
            suspicious_gate_ratio=round(float(suspicious_ratio), 4),
            top_suspicious_gates=top_gates,
            verdict=verdict
        )
