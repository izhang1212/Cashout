"""Shrinkage toward market-implied probabilities.

Until the paper-trading log proves the model beats the market on a leg type,
the market gets the greater weight:

    p_used = (1 - w) * p_model + w * p_market,   w = unproven weight per leg type

This systematically biases the system toward earlier, safer exits -- the
correct direction to be wrong in with real money. `EdgeLedger` is the forward
log that decides when a leg type graduates (w shrinks toward 0).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def shrink(p_model: float, p_market: float, w_market: float) -> float:
    return (1.0 - w_market) * p_model + w_market * p_market


@dataclass
class EdgeLedger:
    """Forward log of model-vs-market disagreements and their outcomes."""
    path: Path
    records: list[dict] = field(default_factory=list)

    def record(self, leg_kind: str, p_model: float, p_market: float, outcome: int) -> None:
        self.records.append({"kind": leg_kind, "p_model": p_model,
                             "p_market": p_market, "outcome": outcome})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.records, f)

    def load(self) -> "EdgeLedger":
        if self.path.exists():
            self.records = json.loads(self.path.read_text())
        return self

    def edge_report(self, leg_kind: str, min_disagreement: float = 0.03) -> dict:
        """Were the model's disagreements with the market right more often than wrong?"""
        rel = [r for r in self.records
               if r["kind"] == leg_kind and abs(r["p_model"] - r["p_market"]) >= min_disagreement]
        if not rel:
            return {"n": 0, "model_right_rate": None, "proven": False}
        right = sum(
            1 for r in rel
            if (r["p_model"] > r["p_market"]) == bool(r["outcome"])
        )
        rate = right / len(rel)
        # crude but honest: require a meaningful sample AND >55% before trusting
        return {"n": len(rel), "model_right_rate": rate,
                "proven": len(rel) >= 100 and rate > 0.55}
