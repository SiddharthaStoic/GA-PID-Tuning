from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.config import SIM_CFG, SimConfig

@dataclass
class Trajectory:
    time: np.ndarray; pitch: np.ndarray; control: np.ndarray; crashed: bool; cfg: SimConfig
    @property
    def max_pitch(self): return float(np.max(np.abs(self.pitch)))

class LinearDeltaModel:
    def run(self, kp: float, ki: float, kd: float, cfg: SimConfig = SIM_CFG) -> Trajectory:
        n = int(cfg.duration / cfg.dt)
        p, v, integr, prev_e = cfg.initial_pitch, 0.0, 0.0, cfg.initial_pitch - cfg.setpoint
        t_arr, p_arr, u_arr = np.zeros(n), np.zeros(n), np.zeros(n)
        crashed = False
        for i in range(n):
            err = cfg.setpoint - p
            integr = np.clip(integr + err * cfg.dt, -cfg.integral_limit, cfg.integral_limit)
            deriv = (err - prev_e) / cfg.dt
            u = np.clip((kp * err) + (ki * integr) + (kd * deriv), -cfg.saturation, cfg.saturation)
            accel = (u * cfg.power_gain - v * cfg.damping) / cfg.inertia
            v += accel * cfg.dt
            p += v * cfg.dt
            t_arr[i], p_arr[i], u_arr[i], prev_e = i * cfg.dt, p, u, err
            if abs(p) > cfg.crash_threshold:
                crashed = True; p_arr[i:], u_arr[i:] = p, u; break
        return Trajectory(t_arr, p_arr, u_arr, crashed, cfg)

class IdentifiedModel:
    """
    Data-driven system model fitted from real telemetry.
    Identifies pitch dynamics: pitch[t+1] = a*pitch[t] + b*control[t] + c
    """
    def __init__(self):
        self._a   = 0.95
        self._b   = 0.01
        self._c   = 0.005
        self.params_ = {"a": self._a, "b": self._b, "c": self._c}
        self.fitted_  = False

    def fit(self, df):
        X = np.column_stack([
            df["pitch_deg"].values[:-1],
            np.zeros(len(df) - 1),
            np.ones(len(df) - 1)
        ])
        y = df["pitch_deg"].values[1:]
        c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self._a, self._b, self._c = float(c[0]), float(c[1]), float(c[2])
        self.params_ = {"a": self._a, "b": self._b, "c": self._c}
        self.fitted_  = True
        return self

    def run(self, kp, ki, kd, cfg=SIM_CFG):
        """Uses identified dynamics to simulate the closed-loop response."""
        n = int(cfg.duration / cfg.dt)
        p, integr, prev_e = cfg.initial_pitch, 0.0, cfg.initial_pitch - cfg.setpoint
        t_arr, p_arr, u_arr = np.zeros(n), np.zeros(n), np.zeros(n)
        crashed = False
        for i in range(n):
            err    = cfg.setpoint - p
            integr = np.clip(integr + err * cfg.dt, -cfg.integral_limit, cfg.integral_limit)
            deriv  = (err - prev_e) / cfg.dt
            u      = np.clip(kp*err + ki*integr + kd*deriv, -cfg.saturation, cfg.saturation)
            p      = self._a * p + self._b * u + self._c
            t_arr[i], p_arr[i], u_arr[i], prev_e = i * cfg.dt, p, u, err
            if abs(p) > cfg.crash_threshold:
                crashed = True; p_arr[i:], u_arr[i:] = p, u; break
        return Trajectory(t_arr, p_arr, u_arr, crashed, cfg)