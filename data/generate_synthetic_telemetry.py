"""
generate_synthetic_telemetry.py
────────────────────────────────
Generates a realistic synthetic IMU telemetry dataset
that mimics ESP32 quadcopter flight recordings.

Output
------
data/raw/test_run_01.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

np.random.seed(42)

n  = 500                        # samples
fs = 100                        # Hz
dt = 1.0 / fs                   # seconds per sample

t = np.linspace(0, n * dt, n)

# ─────────────────────────────────────────────
# Simulate Attitude Angles
# ─────────────────────────────────────────────

# Drone disturbed from hover and stabilizing back
pitch = (
    7.0 * np.exp(-0.8 * t) * np.cos(2 * np.pi * 1.2 * t)
    + np.random.normal(0, 0.3, n)
)

roll = (
    3.0 * np.exp(-1.0 * t) * np.cos(2 * np.pi * 0.8 * t)
    + np.random.normal(0, 0.2, n)
)

yaw = np.cumsum(np.random.normal(0, 0.05, n))

# ─────────────────────────────────────────────
# Derive Gyroscope Signals
# ─────────────────────────────────────────────

gyroX = np.gradient(roll,  t) + np.random.normal(0, 0.5, n)
gyroY = np.gradient(pitch, t) + np.random.normal(0, 0.5, n)
gyroZ = np.gradient(yaw,   t) + np.random.normal(0, 0.3, n)

# ─────────────────────────────────────────────
# Build DataFrame
# ─────────────────────────────────────────────

df_synthetic = pd.DataFrame({
    "dt"        : np.full(n, dt),
    "ax_raw"    : np.random.randint(-1000, 1000, n),
    "ay_raw"    : np.random.randint(-1000, 1000, n),
    "az_raw"    : np.random.randint(15000, 17000, n),
    "gx_raw"    : (gyroX * 131).astype(int),
    "gy_raw"    : (gyroY * 131).astype(int),
    "gz_raw"    : (gyroZ * 131).astype(int),
    "ax_g"      : np.random.uniform(-0.1, 0.1, n),
    "ay_g"      : np.random.uniform(-0.1, 0.1, n),
    "az_g"      : np.random.uniform(0.95, 1.05, n),
    "gyroX_dps" : gyroX,
    "gyroY_dps" : gyroY,
    "gyroZ_dps" : gyroZ,
    "roll_deg"  : roll,
    "pitch_deg" : pitch,
    "yaw_deg"   : yaw,
})

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR  = BASE_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

csv_path = RAW_DIR / "test_run_01.csv"

df_synthetic.to_csv(csv_path, index=False)

print(f"Generated : {n} rows")
print(f"Duration  : {n * dt:.1f} seconds at {fs} Hz")
print(f"Saved to  : {csv_path}")