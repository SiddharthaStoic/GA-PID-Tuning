from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class SimConfig:
    dt              : float = 0.01
    duration        : float = 3.0
    initial_pitch   : float = 25.0
    setpoint        : float = 0.0
    stable_zone_deg : float = 2.0
    crash_threshold : float = 90.0
    inertia         : float = 0.25
    damping         : float = 0.15
    power_gain      : float = 0.80
    saturation      : float = 2000.0
    integral_limit  : float = 10.0

@dataclass(frozen=True)
class GAConfig:
    population_size : int   = 30
    generations     : int   = 15
    elite_count     : int   = 3
    tournament_size : int   = 3
    crossover_rate  : float = 0.80
    mutation_rate   : float = 0.20
    mutation_sigma  : float = 0.30
    mutation_boost  : float = 1.5
    diversity_threshold: float = 0.15
    patience        : int   = 5
    min_delta       : float = 0.001
    seed            : int   = 42

    @property
    def bounds(self) -> dict:
        return {"kp": (0.0, 10.0), "ki": (0.0, 5.0), "kd": (0.0, 5.0)}

@dataclass(frozen=True)
class FitnessWeights:
    iae             : float = 0.40
    overshoot       : float = 0.25
    settling_time   : float = 0.20
    control_effort  : float = 0.10
    crash_penalty   : float = 1.00

SIM_CFG     = SimConfig()
GA_CFG      = GAConfig()
FIT_WEIGHTS = FitnessWeights()

# ─────────────────────────────────────────────
# Paths & Schema (Used by telemetry.py)
# ─────────────────────────────────────────────
RAW_DATA_DIR      = Path(__file__).parent.parent / "data" / "raw"
TELEMETRY_COLUMNS = [
    "dt",
    "pitch_deg",
    "gyroY_dps",
    "ax_g",
    "ay_g",
    "az_g",
]