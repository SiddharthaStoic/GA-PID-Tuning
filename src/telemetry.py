"""
telemetry.py
────────────
Handles loading, validation, and preprocessing of physical telemetry data.
Ensures that raw CSV data is cleaned and formatted correctly for System ID.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import RAW_DATA_DIR, TELEMETRY_COLUMNS

class TelemetryLoader:
    """
    Utility class for ingesting and validating drone logs.
    
    Usage:
        loader = TelemetryLoader()
        df = loader.load("test_run_01.csv")
    """

    def __init__(self, raw_dir: Path = RAW_DATA_DIR):
        self.raw_dir = raw_dir

    def load(self, filename: str) -> pd.DataFrame:
        """Loads a CSV, validates schema, and calculates derived columns."""
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Telemetry file not found: {path}")

        df = pd.read_csv(path)
        
        # 1. Validation
        self._validate_columns(df)
        
        # 2. Preprocessing
        df = self._clean_data(df)
        
        # 3. Frequency Analysis (Information only)
        avg_dt = df["dt"].mean()
        effective_hz = 1.0 / avg_dt if avg_dt > 0 else 0
        print(f"Loaded {filename}: {len(df)} rows @ ~{effective_hz:.1f}Hz")
        
        return df

    def _validate_columns(self, df: pd.DataFrame):
        """Checks if the dataframe contains the required telemetry keys."""
        missing = [col for col in TELEMETRY_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes NaNs, handles sensor noise, and ensures numeric types.
        In Version 2, we normalize units (e.g., ensuring dt is in seconds).
        """
        # Ensure numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Drop corrupted rows
        start_count = len(df)
        df = df.dropna().reset_index(drop=True)
        
        if len(df) < start_count:
            print(f"Warning: Dropped {start_count - len(df)} rows due to NaNs.")

        # If dt is in milliseconds (common in ESP32 logs), convert to seconds
        if df["dt"].max() > 1.0:
            df["dt"] = df["dt"] / 1000.0

        return df

    def get_summary(self, df: pd.DataFrame) -> dict:
        """Returns a snapshot of the telemetry quality (noise, range, etc.)."""
        return {
            "duration": (df["dt"].sum()),
            "pitch_range": (df["pitch_deg"].min(), df["pitch_deg"].max()),
            "gyro_std": df["gyroY_dps"].std(),
            "avg_dt": df["dt"].mean()
        }

def load_and_identify(filename: str, model_class):
    """
    Helper function to bridge telemetry and simulator.
    Loads data and fits a provided model class.
    """
    loader = TelemetryLoader()
    df = loader.load(filename)
    
    # We filter for sections where the drone was actually active/oscillating
    # This prevents the System ID from fitting to static noise
    active_mask = df["gyroY_dps"].abs() > 0.5
    active_df = df[active_mask]
    
    if len(active_df) < 10:
        raise ValueError("Not enough active telemetry data for System Identification.")
        
    model = model_class()
    model.fit(active_df)
    return model