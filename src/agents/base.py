"""Minimal agent primitives (no LangGraph dependency)."""
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class AgentMessage:
    agent: str
    action: str
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    name = "base"
    role = "base agent"

    def can_handle(self, state: Dict[str, Any]) -> bool:
        return True

    def act(self, state: Dict[str, Any]) -> AgentMessage:
        raise NotImplementedError
