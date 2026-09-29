"""
Adaptive Decision Engine for CloudClear AI.
Analyzes scene conditions, temporal change scores, cloud density, and sensor availability
to select the optimal reconstruction strategy.
"""

from enum import Enum
from typing import Dict, Any, Optional
import numpy as np


class ReconstructionStrategy(str, Enum):
    HISTORICAL_DOMINANT = "historical"
    SAR_DOMINANT = "sar"
    ADAPTIVE_FUSION = "adaptive"


class AdaptiveDecisionEngine:
    """
    Intelligent decision engine routing the multi-modal fusion strategy.
    """

    def __init__(
        self,
        stable_threshold: float = 0.25,
        changed_threshold: float = 0.60
    ):
        self.stable_threshold = stable_threshold
        self.changed_threshold = changed_threshold

    def evaluate_strategy(
        self,
        cloud_pct: float,
        change_score: float,
        changed_area_pct: float,
        has_sar: bool = True,
        has_historical: bool = True
    ) -> Dict[str, Any]:
        """
        Determines the optimal reconstruction strategy.

        Returns:
            {
                "strategy": str ("historical", "sar", or "adaptive"),
                "recommended_candidate": str ("C1", "C2", or "C3"),
                "rationale": str,
                "weights": {"historical": float, "sar": float, "optical": float},
                "confidence_prior": float
            }
        """
        if not has_historical and not has_sar:
            raise ValueError("Cannot reconstruct: Neither historical nor SAR data is available.")

        if not has_historical:
            return {
                "strategy": ReconstructionStrategy.SAR_DOMINANT.value,
                "recommended_candidate": "C2",
                "rationale": "Historical archive unavailable; routed exclusively to SAR structural reconstruction.",
                "weights": {"historical": 0.0, "sar": 0.85, "optical": 0.15},
                "confidence_prior": 0.72
            }

        if not has_sar:
            return {
                "strategy": ReconstructionStrategy.HISTORICAL_DOMINANT.value,
                "recommended_candidate": "C1",
                "rationale": "SAR data unavailable; routed to historical temporal synthesis.",
                "weights": {"historical": 0.85, "sar": 0.0, "optical": 0.15},
                "confidence_prior": 0.80
            }

        # Both Historical and SAR available
        if change_score < self.stable_threshold and changed_area_pct < 15.0:
            strategy = ReconstructionStrategy.HISTORICAL_DOMINANT.value
            cand = "C1"
            rationale = (
                f"Terrain is stable (Change Score: {change_score:.2f}, Changed Area: {changed_area_pct:.1f}%). "
                "Historical reference provides high-fidelity spectral reconstruction."
            )
            weights = {"historical": 0.75, "sar": 0.15, "optical": 0.10}
            conf = 0.92

        elif change_score > self.changed_threshold or changed_area_pct > 50.0:
            strategy = ReconstructionStrategy.SAR_DOMINANT.value
            cand = "C2"
            rationale = (
                f"Significant surface change detected (Change Score: {change_score:.2f}, Changed Area: {changed_area_pct:.1f}%). "
                "SAR radar backscatter prioritizes real-time surface structure over outdated historical imagery."
            )
            weights = {"historical": 0.20, "sar": 0.70, "optical": 0.10}
            conf = 0.85

        else:
            strategy = ReconstructionStrategy.ADAPTIVE_FUSION.value
            cand = "C3"
            rationale = (
                f"Mixed surface conditions (Change Score: {change_score:.2f}, Changed Area: {changed_area_pct:.1f}%). "
                "Adaptive Cross-Attention Fusion balances historical spectral fidelity with SAR structural penetration."
            )
            weights = {"historical": 0.45, "sar": 0.45, "optical": 0.10}
            conf = 0.90

        return {
            "strategy": strategy,
            "recommended_candidate": cand,
            "rationale": rationale,
            "weights": weights,
            "confidence_prior": conf
        }
