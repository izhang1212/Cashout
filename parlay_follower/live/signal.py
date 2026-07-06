"""HOLD/SELL signal object + one-line human explanation."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Signal:
    action: str                 # "HOLD" | "SELL"
    margin: float               # executable_bid - continuation_value
    executable_bid: float
    fair_value: float
    continuation_value: float
    ensemble_votes_sell: int = 0
    ensemble_size: int = 0
    reason: str = ""
    ts: float = field(default_factory=time.time)

    def line(self) -> str:
        core = (f"[{self.action}] bid=${self.executable_bid:.3f} "
                f"fair=${self.fair_value:.3f} cont=${self.continuation_value:.3f} "
                f"margin=${self.margin:+.3f}")
        if self.ensemble_size:
            core += f" ensemble={self.ensemble_votes_sell}/{self.ensemble_size} sell"
        if self.reason:
            core += f" | {self.reason}"
        return core
