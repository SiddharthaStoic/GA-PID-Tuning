from __future__ import annotations
from src.config import FIT_WEIGHTS, SIM_CFG, FitnessWeights
from src.simulator import Trajectory
from src.features import FeatureVector, extract_features


class FitnessEvaluator:
    def __init__(self, cfg: FitnessWeights = FIT_WEIGHTS):
        self.cfg = cfg

    def evaluate(self, trj: Trajectory) -> float:
        if trj.crashed:
            return self.cfg.crash_penalty
        fv = extract_features(trj)
        worst_iae = SIM_CFG.crash_threshold * SIM_CFG.duration
        fitness = (
            self.cfg.iae            * min(fv.iae / worst_iae, 1.0) +
            self.cfg.overshoot      * min(fv.overshoot / SIM_CFG.crash_threshold, 1.0) +
            self.cfg.settling_time  * min(fv.settling_time / SIM_CFG.duration, 1.0) +
            self.cfg.control_effort * min(fv.control_effort / 1000.0, 1.0)
        )
        return round(float(fitness), 6)

    def breakdown(self, trj: Trajectory) -> dict:
        """Returns a detailed breakdown of each fitness component."""
        if trj.crashed:
            return {"crashed": True, "total": self.cfg.crash_penalty}

        fv = extract_features(trj)
        worst_iae = SIM_CFG.crash_threshold * SIM_CFG.duration

        iae_score       = self.cfg.iae            * min(fv.iae / worst_iae, 1.0)
        overshoot_score = self.cfg.overshoot      * min(fv.overshoot / SIM_CFG.crash_threshold, 1.0)
        settling_score  = self.cfg.settling_time  * min(fv.settling_time / SIM_CFG.duration, 1.0)
        effort_score    = self.cfg.control_effort * min(fv.control_effort / 1000.0, 1.0)
        total           = iae_score + overshoot_score + settling_score + effort_score

        return {
            "iae":            round(iae_score, 6),
            "overshoot":      round(overshoot_score, 6),
            "settling_time":  round(settling_score, 6),
            "control_effort": round(effort_score, 6),
            "total":          round(total, 6),
            "crashed":        False,
            "raw": {
                "iae":           round(fv.iae, 4),
                "overshoot":     round(fv.overshoot, 4),
                "settling_time": round(fv.settling_time, 4),
                "control_effort":round(fv.control_effort, 4),
            }
        }