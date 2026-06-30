"""
processing.py
─────────────
Signal processing utilities for drone telemetry.

Responsibilities
----------------
• Remove sensor noise
• Smooth IMU measurements
• Detect outliers
• Resample telemetry
• Prepare data for System Identification

This module NEVER computes fitness and NEVER performs optimization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class SignalProcessor:
    """
    Collection of signal processing methods.

    Example
    -------
    processor = SignalProcessor()

    df["pitch_deg"] = processor.lowpass(
        df["pitch_deg"],
        cutoff=5,
        fs=50
    )
    """

    @staticmethod
    def moving_average(signal, window: int = 5):
        """
        Simple Moving Average filter.
        Useful when SciPy is unavailable.
        """

        signal = np.asarray(signal)

        if window <= 1:
            return signal

        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode="same")

    @staticmethod
    def butterworth_lowpass(signal,
                            cutoff: float,
                            fs: float,
                            order: int = 2):
        """
        Butterworth Low-Pass Filter.

        Parameters
        ----------
        cutoff : Hz
        fs : Sampling frequency
        order : Filter order
        """

        if not SCIPY_AVAILABLE:
            raise ImportError(
                "SciPy is required for Butterworth filtering."
            )

        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist

        b, a = butter(
            order,
            normal_cutoff,
            btype="low",
            analog=False
        )

        return filtfilt(b, a, signal)

    @staticmethod
    def remove_outliers(signal,
                        z_threshold: float = 3.0):
        """
        Removes statistical outliers using Z-score.
        """

        signal = np.asarray(signal)

        mean = np.mean(signal)
        std = np.std(signal)

        if std == 0:
            return signal

        z = np.abs((signal - mean) / std)

        filtered = signal.copy()
        filtered[z > z_threshold] = mean

        return filtered

    @staticmethod
    def estimate_noise(signal):
        """
        Estimate sensor noise (standard deviation).
        """

        signal = np.asarray(signal)

        return {
            "mean": float(np.mean(signal)),
            "std": float(np.std(signal)),
            "variance": float(np.var(signal))
        }

    @staticmethod
    def normalize(signal):
        """
        Normalize to [0,1].
        """

        signal = np.asarray(signal)

        minimum = signal.min()
        maximum = signal.max()

        if maximum == minimum:
            return np.zeros_like(signal)

        return (signal - minimum) / (maximum - minimum)

    @staticmethod
    def standardize(signal):
        """
        Standard score normalization.
        """

        signal = np.asarray(signal)

        std = signal.std()

        if std == 0:
            return np.zeros_like(signal)

        return (signal - signal.mean()) / std

    @staticmethod
    def resample(df: pd.DataFrame,
                 target_dt: float):
        """
        Resample telemetry to a uniform timestep.

        Useful when ESP32 timestamps are irregular.
        """

        time = np.cumsum(df["dt"].values)

        new_time = np.arange(
            time.min(),
            time.max(),
            target_dt
        )

        new_df = pd.DataFrame({
            "time": new_time
        })

        for column in df.columns:

            if column == "dt":
                continue

            new_df[column] = np.interp(
                new_time,
                time,
                df[column]
            )

        new_df["dt"] = target_dt

        return new_df

    @staticmethod
    def process_dataframe(df: pd.DataFrame,
                          sampling_rate: float = 50.0):
        """
        Complete preprocessing pipeline.

        Returns cleaned DataFrame.
        """

        processed = df.copy()

        imu_columns = [
            "pitch_deg",
            "roll_deg",
            "yaw_deg",
            "gyroX_dps",
            "gyroY_dps",
            "gyroZ_dps"
        ]

        for column in imu_columns:

            if column not in processed.columns:
                continue

            signal = processed[column].values

            signal = SignalProcessor.remove_outliers(signal)

            if SCIPY_AVAILABLE:
                signal = SignalProcessor.butterworth_lowpass(
                    signal,
                    cutoff=5,
                    fs=sampling_rate
                )
            else:
                signal = SignalProcessor.moving_average(
                    signal,
                    window=5
                )

            processed[column] = signal

        return processed