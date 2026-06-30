"""
optimization/base.py
────────────────────
Defines the abstract interface for all PID optimizers.
Whether it's GA, PSO, or Bayesian, they all follow this contract.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import time

@dataclass
class PIDCandidate:
    """A single chromosome/particle representing a PID configuration."""
    kp: float
    ki: float
    kd: float
    fitness: float = float('inf')
    
    # Metadata for tracking
    generation: int = 0
    id: str = field(default_factory=lambda: hex(int(time.time() * 1e7))[-6:])

    def to_array(self) -> np.ndarray:
        return np.array([self.kp, self.ki, self.kd])

@dataclass
class OptimizationHistory:
    """Container for tracking the evolution process over time."""
    best_scores: list[float] = field(default_factory=list)
    mean_scores: list[float] = field(default_factory=list)
    best_candidates: list[PIDCandidate] = field(default_factory=list)
    diversity_scores: list[float] = field(default_factory=list)

class BaseOptimizer(ABC):
    """
    Abstract Base Class for all optimizers.
    Forces a consistent API: .optimize() and .step()
    """
    
    def __init__(self, simulator, evaluator, config):
        self.sim = simulator
        self.eval = evaluator
        self.cfg = config
        self.history = OptimizationHistory()
    
    @abstractmethod
    def optimize(self) -> OptimizationHistory:
        """The main loop that runs the full optimization process."""
        pass

    def evaluate_candidate(self, cand: PIDCandidate) -> float:
        """Simulates and scores a candidate. Centralizes evaluation logic."""
        trj = self.sim.run(cand.kp, cand.ki, cand.kd)
        score = self.eval.evaluate(trj)
        cand.fitness = score
        return score