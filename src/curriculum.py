"""Curriculum scheduler and CBS oracle annealing for MAPF training."""

from __future__ import annotations

from collections import deque
from typing import Optional

from .maps import DIFFICULTY_LEVELS, MapConfig


class CBSAnnealer:
    """
    Three-phase CBS oracle weight schedule:

    Phase A  [0 … warmup_steps]          weight = 1.0  (follow CBS fully)
    Phase B  [warmup_steps … anneal_end] weight 1.0 → 0.0  (linearly)
    Phase C  [anneal_end … ∞]            weight = 0.0  (pure RL, no oracle)

    Call .step() once per environment step to advance the counter.
    Read .weight to get the current CBS shaping multiplier.
    """

    def __init__(self, warmup_steps: int = 50_000, anneal_end: int = 200_000):
        self.warmup_steps = warmup_steps
        self.anneal_end   = anneal_end
        self._step        = 0

    def step(self) -> float:
        """Advance one global step. Returns the current weight."""
        self._step += 1
        return self.weight

    @property
    def weight(self) -> float:
        if self._step <= self.warmup_steps:
            return 1.0
        if self._step >= self.anneal_end:
            return 0.0
        progress = (self._step - self.warmup_steps) / (self.anneal_end - self.warmup_steps)
        return float(1.0 - progress)

    @property
    def phase(self) -> str:
        if self._step <= self.warmup_steps:
            return "A"
        if self._step >= self.anneal_end:
            return "C"
        return "B"

    @property
    def global_step(self) -> int:
        return self._step


class DifficultyScheduler:
    """
    Auto-curriculum over four difficulty levels.

    Tracks a sliding window of episode success rates.
    Advances the difficulty when the window mean exceeds *advance_threshold*,
    and regresses when it drops below *regress_threshold*.
    The window is cleared on every transition to give the agent a fresh start.

    Levels (in order): easy → medium → hard → expert
    """

    LEVELS = ["easy", "medium", "hard", "expert"]

    def __init__(
        self,
        start_level: str = "easy",
        advance_threshold: float = 0.80,
        regress_threshold: float = 0.40,
        window: int = 100,
    ):
        assert start_level in self.LEVELS, f"Unknown level: {start_level}"
        self._idx               = self.LEVELS.index(start_level)
        self.advance_threshold  = advance_threshold
        self.regress_threshold  = regress_threshold
        self._history: deque[float] = deque(maxlen=window)

    def record(self, success_rate: float) -> Optional[str]:
        """
        Record one episode's success rate (0–1).
        Returns the new level name if difficulty changed, else None.
        """
        self._history.append(success_rate)
        min_samples = max(10, self._history.maxlen // 4)
        if len(self._history) < min_samples:
            return None

        mean = sum(self._history) / len(self._history)

        if mean >= self.advance_threshold and self._idx < len(self.LEVELS) - 1:
            self._idx += 1
            self._history.clear()
            return self.current_level

        if mean < self.regress_threshold and self._idx > 0:
            self._idx -= 1
            self._history.clear()
            return self.current_level

        return None

    @property
    def current_level(self) -> str:
        return self.LEVELS[self._idx]

    @property
    def current_config(self) -> MapConfig:
        return DIFFICULTY_LEVELS[self.current_level]

    @property
    def success_rate(self) -> float:
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)
