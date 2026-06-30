import subprocess
from pathlib import Path
from src.config import GA_CFG
from src.simulator import LinearDeltaModel
from src.fitness import FitnessEvaluator
from src.optimization.genetic import GeneticOptimizer
from src.database import ExperimentDB
from src.visualization.plots import Visualizer

def main():
    sim, ev, db, viz = LinearDeltaModel(), FitnessEvaluator(), ExperimentDB(), Visualizer()
    opt = GeneticOptimizer(sim, ev, GA_CFG)
    print("Running GA Optimization...")
    history = opt.optimize()
    best = opt.best
    print(f"\nResult: Kp={best.kp:.2f} Ki={best.ki:.2f} Kd={best.kd:.2f} | Fitness: {best.fitness:.4f}")
    db.save_run(best.kp, best.ki, best.kd, best.fitness, best.generation)
    viz.plot_evolution(history.best_scores, history.mean_scores, history.diversity_scores)
    viz.plot_trajectory(sim.run(best.kp, best.ki, best.kd), best.kp, best.ki, best.kd)

if __name__ == "__main__": main()