"""
optimization/genetic.py
───────────────────────
Genetic Algorithm optimizer for PID tuning.

Improvements over original ga_optimizer.py
───────────────────────────────────────────
• Elitism          — top N candidates survive unchanged
• Tournament       — selection pressure without fitness scaling issues
• Adaptive mutation — boosts sigma when diversity collapses
• Early stopping   — patience-based convergence detection
• Reproducibility  — seeded RNG via GAConfig.seed
• Typed candidates — PIDCandidate dataclass, not raw dicts
• Clean history    — OptimizationHistory, not a list of tuples
"""

from __future__ import annotations

import numpy as np
from src.config import GA_CFG, GAConfig
from src.optimization.base import BaseOptimizer, PIDCandidate, OptimizationHistory
from src.optimization.utils import (
    tournament_selection,
    calculate_diversity,
    clamp_bounds,
)


class GeneticOptimizer(BaseOptimizer):
    """
    Genetic Algorithm for PID coefficient optimization.

    Usage
    -----
        sim       = LinearDeltaModel()
        evaluator = FitnessEvaluator()
        ga        = GeneticOptimizer(sim, evaluator, GA_CFG)
        history   = ga.optimize()
    """

    def __init__(self, simulator, evaluator, config: GAConfig = GA_CFG):
        super().__init__(simulator, evaluator, config)
        self.rng = np.random.default_rng(config.seed)

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def optimize(self) -> OptimizationHistory:
        """Run the full GA evolution loop."""

        population = self._initialize_population()
        self._evaluate_population(population)

        best_score   = float("inf")
        patience_ctr = 0

        for gen in range(self.cfg.generations):

            # ── Elitism ──────────────────────
            population.sort(key=lambda c: c.fitness)
            elites = [self._clone(c) for c in population[: self.cfg.elite_count]]

            # ── Diversity check ──────────────
            diversity = calculate_diversity(population)
            sigma = (
                self.cfg.mutation_sigma * self.cfg.mutation_boost
                if diversity < self.cfg.diversity_threshold
                else self.cfg.mutation_sigma
            )

            # ── Breed next generation ────────
            offspring = elites[:]
            while len(offspring) < self.cfg.population_size:
                parent_a = tournament_selection(population, self.cfg.tournament_size)
                parent_b = tournament_selection(population, self.cfg.tournament_size)
                child    = self._crossover(parent_a, parent_b, gen)
                child    = self._mutate(child, sigma)
                offspring.append(child)

            # ── Evaluate offspring ───────────
            self._evaluate_population(offspring[self.cfg.elite_count :])
            population = offspring

            # ── Record history ───────────────
            population.sort(key=lambda c: c.fitness)
            gen_best  = population[0].fitness
            gen_mean  = float(np.mean([c.fitness for c in population]))

            self.history.best_scores.append(gen_best)
            self.history.mean_scores.append(gen_mean)
            self.history.best_candidates.append(self._clone(population[0]))
            self.history.diversity_scores.append(diversity)

            print(
                f"Gen {gen + 1:>3}/{self.cfg.generations} | "
                f"Best: {gen_best:.6f} | "
                f"Mean: {gen_mean:.6f} | "
                f"Diversity: {diversity:.4f} | "
                f"Sigma: {sigma:.3f}"
            )

            # ── Early stopping ───────────────
            if best_score - gen_best > self.cfg.min_delta:
                best_score   = gen_best
                patience_ctr = 0
            else:
                patience_ctr += 1

            if patience_ctr >= self.cfg.patience:
                print(f"\nEarly stopping at generation {gen + 1}.")
                print(f"No improvement for {self.cfg.patience} consecutive generations.")
                break

        return self.history

    @property
    def best(self) -> PIDCandidate:
        """Returns the best candidate found across all generations."""
        if not self.history.best_candidates:
            raise RuntimeError("Optimizer has not been run yet.")
        return min(self.history.best_candidates, key=lambda c: c.fitness)

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _initialize_population(self) -> list[PIDCandidate]:
        """Uniform random initialization within bounds."""
        population = []
        for _ in range(self.cfg.population_size):
            kp = self.rng.uniform(*self.cfg.bounds["kp"])
            ki = self.rng.uniform(*self.cfg.bounds["ki"])
            kd = self.rng.uniform(*self.cfg.bounds["kd"])
            population.append(PIDCandidate(kp=kp, ki=ki, kd=kd))
        return population

    def _evaluate_population(self, population: list[PIDCandidate]):
        """Evaluates fitness for every candidate in place."""
        for candidate in population:
            self.evaluate_candidate(candidate)

    def _crossover(self,
                   parent_a: PIDCandidate,
                   parent_b: PIDCandidate,
                   generation: int) -> PIDCandidate:
        """
        Uniform crossover.
        Each gene is independently drawn from either parent
        with probability crossover_rate.
        """
        if self.rng.random() > self.cfg.crossover_rate:
            return self._clone(parent_a)

        genes_a = parent_a.to_array()
        genes_b = parent_b.to_array()
        mask    = self.rng.random(3) < 0.5
        child_genes = np.where(mask, genes_a, genes_b)

        return PIDCandidate(
            kp=child_genes[0],
            ki=child_genes[1],
            kd=child_genes[2],
            generation=generation,
        )

    def _mutate(self, candidate: PIDCandidate, sigma: float) -> PIDCandidate:
        """
        Gaussian mutation with per-gene probability.
        Clamps result to bounds after mutation.
        """
        genes = candidate.to_array()
        keys  = ["kp", "ki", "kd"]

        for i, key in enumerate(keys):
            if self.rng.random() < self.cfg.mutation_rate:
                genes[i] += self.rng.normal(0, sigma)
                genes[i]  = clamp_bounds(genes[i], self.cfg.bounds[key])

        candidate.kp = genes[0]
        candidate.ki = genes[1]
        candidate.kd = genes[2]
        return candidate

    @staticmethod
    def _clone(candidate: PIDCandidate) -> PIDCandidate:
        """Deep copy of a candidate."""
        return PIDCandidate(
            kp=candidate.kp,
            ki=candidate.ki,
            kd=candidate.kd,
            fitness=candidate.fitness,
            generation=candidate.generation,
        )