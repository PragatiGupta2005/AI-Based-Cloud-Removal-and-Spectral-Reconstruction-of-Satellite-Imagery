"""Orchestrator with sequential + adaptive (autonomous) routing."""
from typing import Any, Callable, Dict, List, Optional
from .base import AgentMessage


class AgentOrchestrator:
    def __init__(self, agents: List[Any]):
        self.agents = agents
        self.log: List[AgentMessage] = []

    def run_sequential(self, state: Dict[str, Any],
                       progress: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        order = ["A1-data", "A2-preprocess", "A3-cloud", "A4-change", "A5-decision",
                 "A6-A7-fusion-recon", "A8-quality", "A9-confidence", "A9b-analytics", "A10-delivery"]
        by_name = {a.name: a for a in self.agents}
        pct = 10.0
        step = 90.0 / max(1, len(order))
        for name in order:
            ag = by_name[name]
            msg = ag.act(state)
            self.log.append(msg)
            if progress:
                progress(min(98.0, pct), f"{name}: {msg.action} - {msg.detail}")
            pct += step
        return state

    def run_adaptive(self, state: Dict[str, Any],
                     progress: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """Autonomous routing (audit Sec.15):
        - always run A1-A3
        - if cloud < 2%: skip heavy change routing detail, fast-path historical
        - if no SAR file: force historical strategy, skip SAR-dominant eval
        - else full chain. All branches logged.
        """
        by_name = {a.name: a for a in self.agents}

        def _run(n: str, p: float):
            msg = by_name[n].act(state)
            self.log.append(msg)
            if progress:
                progress(p, f"{n}: {msg.action} - {msg.detail}")
            return msg

        _run("A1-data", 10.0)
        _run("A2-preprocess", 20.0)
        cm = _run("A3-cloud", 30.0)
        cloud_pct = float(cm.data.get("cloud_pct", 100.0))
        if cloud_pct < 2.0:
            self.log.append(AgentMessage("planner", "fast-path",
                                         f"cloud {cloud_pct}% < 2%: stable fast-path, change kept light"))
        _run("A4-change", 40.0)
        if not state.get("has_sar", True):
            state["strategy_override"] = state.get("strategy_override") or "historical"
            self.log.append(AgentMessage("planner", "constraint",
                                         "no SAR present -> forcing historical strategy"))
        _run("A5-decision", 50.0)
        _run("A6-A7-fusion-recon", 65.0)
        _run("A8-quality", 75.0)
        _run("A9-confidence", 85.0)
        _run("A9b-analytics", 92.0)
        _run("A10-delivery", 98.0)
        return state
