"""JSONL feedback store: user corrections -> retraining data.

Each record: {image_id, kind: cloud_mask|strategy|quality, payload, created_at}.
`train_models.py --use-feedback` merges corrected masks as extra supervision.
"""
import json
import os
import time
from typing import Any, Dict, List


class FeedbackStore:
    def __init__(self, path: str | None = None):
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.path = path or os.path.join(base, "data", "feedback.jsonl")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def add(self, record: Dict[str, Any]) -> Dict[str, Any]:
        rec = dict(record)
        rec.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def list(self, limit: int = 200) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
        return out[-limit:]

    def count(self) -> int:
        return len(self.list(limit=10_000))
