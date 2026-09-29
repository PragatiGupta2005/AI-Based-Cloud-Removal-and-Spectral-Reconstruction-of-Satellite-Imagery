"""True multi-agent orchestration (audit Sec.14/15).

Each pipeline stage is wrapped as an autonomous Agent with:
  - name / role
  - `can_handle(state)` precondition
  - `act(state)` execution returning a message log

`AgentOrchestrator` supports:
  - mode="sequential": fixed A1->A10 order (legacy behavior preserved)
  - mode="adaptive": dynamic routing — e.g. skip SAR-heavy C2 when no SAR,
    prefer C1 fast-path when stable & low cloud, escalate to full fusion
    otherwise. Every decision is logged for explainability.
"""
from .base import BaseAgent, AgentMessage
from .agents import (
    DataRetrievalAgent, PreprocessingAgent, CloudDetectionAgent,
    ChangeDetectionAgent, DecisionAgent, FusionReconstructionAgent,
    QualityAgent, ConfidenceAgent, AnalyticsAgent, DeliveryAgent,
)
from .orchestrator import AgentOrchestrator

__all__ = [
    "BaseAgent", "AgentMessage", "AgentOrchestrator",
    "DataRetrievalAgent", "PreprocessingAgent", "CloudDetectionAgent",
    "ChangeDetectionAgent", "DecisionAgent", "FusionReconstructionAgent",
    "QualityAgent", "ConfidenceAgent", "AnalyticsAgent", "DeliveryAgent",
]
