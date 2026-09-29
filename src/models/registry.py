"""Pretrained-model registry (audit Sec.13).

Every AI component declares: architecture, weights path, training data,
validation status, and whether inference is learned / rule-based / hybrid.
Pipeline and API expose this so docs never over-claim a 'pretrained model'.
"""
import json
import os
from typing import Any, Dict, List

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGISTRY_PATH = os.path.join(_BASE, "models", "MODEL_REGISTRY.json")


def _exists(rel: str) -> bool:
    return os.path.exists(os.path.join(_BASE, rel))


def get_registry() -> Dict[str, Any]:
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"models": []}


def refresh_registry() -> Dict[str, Any]:
    """Recomputes weights availability from disk (no heavy imports)."""
    reg = get_registry()
    for m in reg.get("models", []):
        wp = m.get("weights", "")
        m["weights_available"] = bool(wp) and os.path.exists(
            wp if os.path.isabs(wp) else os.path.join(_BASE, wp)
        )
    return reg


def model_status(name: str) -> Dict[str, Any]:
    for m in get_registry().get("models", []):
        if m.get("name") == name:
            return m
    return {}
