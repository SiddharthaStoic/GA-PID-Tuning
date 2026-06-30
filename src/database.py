from __future__ import annotations
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
import pandas as pd


class ExperimentDB:
    def __init__(self, db_path: str | Path = "data/experiments.db"):
        self.db_path = Path(__file__).parent.parent / db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Ensures the table exists with the correct schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id          TEXT PRIMARY KEY,
                    timestamp   TEXT,
                    kp          REAL,
                    ki          REAL,
                    kd          REAL,
                    fitness     REAL,
                    generation  INTEGER,
                    model_type  TEXT,
                    notes       TEXT
                )
            """)
            # Auto-migrate old tables missing the notes column
            existing = [r[1] for r in conn.execute("PRAGMA table_info(runs)")]
            if "notes" not in existing:
                conn.execute("ALTER TABLE runs ADD COLUMN notes TEXT")

    def save_run(self, kp=None, ki=None, kd=None, fitness=None,
                 generation=0, model_type="linear", notes="", candidate=None):
        """Saves an optimization run. Supports both raw values and PIDCandidate objects."""
        if candidate is not None:
            kp         = candidate.kp
            ki         = candidate.ki
            kd         = candidate.kd
            fitness    = candidate.fitness
            generation = candidate.generation

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO runs
                   (id, timestamp, kp, ki, kd, fitness, generation, model_type, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    datetime.now().isoformat(),
                    float(kp),
                    float(ki),
                    float(kd),
                    float(fitness),
                    int(generation),
                    str(model_type),
                    str(notes)
                )
            )

    def get_history(self) -> pd.DataFrame:
        """Returns all historical runs as a DataFrame, newest first."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM runs ORDER BY timestamp DESC", conn
            )

    def get_best(self) -> pd.Series:
        """Returns the single best run (lowest fitness) ever recorded."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM runs ORDER BY fitness ASC LIMIT 1", conn
            )
        if df.empty:
            raise ValueError("No experiments found in the database.")
        return df.iloc[0]

    def get_best_ever(self) -> pd.Series:
        """Alias for get_best(). Returns the best run ever recorded."""
        return self.get_best()

    def get_runs_by_model(self, model_type: str) -> pd.DataFrame:
        """Returns all runs filtered by model type."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM runs WHERE model_type = ? ORDER BY fitness ASC",
                conn, params=(model_type,)
            )

    def clear(self):
        """Deletes all records. Useful for resetting between experiments."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM runs")
        print("Database cleared.")