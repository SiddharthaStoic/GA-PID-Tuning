"""
features.py
───────────
Transforms a raw Trajectory into a structured FeatureVector.
The FitnessEvaluator consumes this vector to calculate a single scalar score.
Separating features from fitness allows for cleaner multi-objective optimization.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.simulator import Trajectory

@dataclass(frozen=True)
class FeatureVector:
    """Quantitative performance metrics extracted from a simulation run."""
    iae:            float  # Integral Absolute Error
    overshoot:      float  # Maximum degrees beyond stable_zone_deg
    settling_time:  float  # Time (sec) to stay within ±settling_threshold_deg
    control_effort: float  # Proxy for energy/actuator wear (integral of u²)
    is_stable:      bool   # True if it didn't crash

def extract_features(trj: Trajectory) -> FeatureVector:
    """Analyzes a Trajectory to build a FeatureVector."""
    cfg = trj.cfg
    
    # 1. IAE (Integral Absolute Error)
    iae = np.sum(np.abs(trj.pitch)) * trj.cfg.dt
    
    # 2. Overshoot
    # We only penalize degrees *above* the stable_zone_deg threshold
    max_p = np.max(np.abs(trj.pitch))
    overshoot = max(0.0, max_p - cfg.stable_zone_deg)
    
    # 3. Settling Time
    # Defined as the last time |pitch| was outside the threshold
    threshold = 1.0  # degrees
    outside = np.where(np.abs(trj.pitch) > threshold)[0]
    if len(outside) == 0:
        settling_time = 0.0
    else:
        last_index = outside[-1]
        settling_time = trj.time[last_index]
        
    # 4. Control Effort
    # Sum of squared control inputs (penalizes aggressive oscillations)
    control_effort = np.sum(np.square(trj.control)) * trj.cfg.dt
    
    return FeatureVector(
        iae=float(iae),
        overshoot=float(overshoot),
        settling_time=float(settling_time),
        control_effort=float(control_effort),
        is_stable=not trj.crashed
    )