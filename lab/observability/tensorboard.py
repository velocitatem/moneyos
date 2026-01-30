"""
TensorBoard logging helpers for simulator rollouts.

Provides a thin wrapper around tensorboardX.SummaryWriter that understands
StepResult structures and rollout summaries so experiments can emit rich
telemetry with minimal wiring.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping, TYPE_CHECKING

import numpy as np

try:
    from tensorboardX import SummaryWriter
except ImportError as exc:  # pragma: no cover - dependency enforced in requirements
    raise RuntimeError(
        "tensorboardX is required for TensorBoard logging. "
        "Install optional observability extras or add tensorboardX to your environment."
    ) from exc

if TYPE_CHECKING:
    from lab.experiments.eval import RolloutResult
    from ..outlet.types import StepResult


def _ensure_path(path: str | Path) -> Path:
    target = Path(path).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    return target


class TensorBoardLogger:
    """Domain-aware TensorBoard logger.

    Args:
        log_dir: Base directory for TensorBoard runs.
        run_name: Optional subdirectory name; defaults to UTC timestamp.
        flush_secs: How often to flush data to disk.
    """

    def __init__(self, log_dir: str | Path, run_name: str | None = None, flush_secs: int = 10):
        base = _ensure_path(log_dir)
        self._run_name = run_name or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self.log_dir = _ensure_path(base / self._run_name)
        self.writer = SummaryWriter(logdir=str(self.log_dir), flush_secs=flush_secs)
        self._cumulative_reward: MutableMapping[str, float] = defaultdict(float)

    def log_rollout_step(self, step: int, result: "StepResult", tag: str | None = None) -> None:
        """Log per-step reward, metrics, and contamination."""
        base = tag or "rollout"
        metrics = result.metrics

        scalars: Mapping[str, float] = {
            "reward": float(result.reward),
            "pnl": float(metrics.pnl),
            "revenue": float(metrics.revenue),
            "cost": float(metrics.cost),
            "units_traded": float(metrics.units_traded),
            "position_cost": float(metrics.position_cost),
            "lost_opportunity": float(metrics.lost_opportunity),
            "spread_capture": float(metrics.spread_capture),
            "volatility": float(metrics.volatility),
            "conversion": float(metrics.conversion),
        }

        for name, value in scalars.items():
            self.writer.add_scalar(f"{base}/{name}", value, step)

        self._cumulative_reward[base] += float(result.reward)
        self.writer.add_scalar(f"{base}/cumulative_reward", self._cumulative_reward[base], step)

        contamination = getattr(result.hidden, "contamination", None)
        if contamination is not None:
            self.writer.add_scalar(f"{base}/contamination", float(contamination), step)

        per_instrument = getattr(metrics, "per_instrument", {}) or {}
        for key, values in per_instrument.items():
            arr = np.asarray(values, dtype=np.float32)
            if arr.size == 0:
                continue
            self.writer.add_scalar(f"{base}/per_instrument/{key}_mean", float(np.mean(arr)), step)
            self.writer.add_scalar(f"{base}/per_instrument/{key}_sum", float(np.sum(arr)), step)

    def log_rollout_summary(self, rollout: "RolloutResult", tag: str | None = None) -> None:
        """Log aggregate reward and conversion statistics after a rollout."""
        base = (tag or "rollout") + "/summary"
        horizon = len(rollout.rewards)
        self.writer.add_scalar(f"{base}/total_reward", float(rollout.total_reward), horizon)
        self.writer.add_scalar(f"{base}/total_pnl", float(rollout.total_pnl), horizon)
        self.writer.add_scalar(f"{base}/avg_conversion", float(rollout.avg_conversion), horizon)

    def log_policy_summary(self, name: str, stats: Mapping[str, float], step: int | None = None,
                           prefix: str = "policies") -> None:
        """Log aggregated policy comparison statistics."""
        base = f"{prefix}/{name}"
        final_step = step if step is not None else 0
        for key, value in stats.items():
            self.writer.add_scalar(f"{base}/{key}", float(value), final_step)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def create_logger(log_dir: str | Path, run_name: str | None = None, flush_secs: int = 10) -> TensorBoardLogger:
    """Factory helper mirroring TensorBoardLogger constructor."""
    return TensorBoardLogger(log_dir=log_dir, run_name=run_name, flush_secs=flush_secs)
