"""
optimization/utils.py
─────────────────────
Shared math and selection logic used by multiple optimizers.
"""

import numpy as np
from src.optimization.base import PIDCandidate

def tournament_selection(population: list[PIDCandidate], k: int = 3) -> PIDCandidate:
    """Selects the best individual from a random subset of size k."""
    idx = np.random.choice(len(population), k, replace=False)
    competitors = [population[i] for i in idx]
    return min(competitors, key=lambda c: c.fitness)

def calculate_diversity(population: list[PIDCandidate]) -> float:
    """
    Calculates the standard deviation of genes in the population.
    Low diversity signals premature convergence.
    """
    genes = np.array([[c.kp, c.ki, c.kd] for c in population])
    return float(np.mean(np.std(genes, axis=0)))

def clamp_bounds(val: float, bounds: tuple[float, float]) -> float:
    """Ensures a gene stays within physical limits."""
    return max(bounds[0], min(val, bounds[1]))