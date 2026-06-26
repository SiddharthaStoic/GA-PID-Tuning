import os
from pathlib import Path

# Визначаємо кореневу папку проєкту (піднімаємося на рівень вище від папки src)
BASE_DIR = Path(__file__).resolve().parent.parent

# Шляхи до папок з даними
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")

# Шлях до бази даних SQLite
DB_PATH = os.path.join(PROCESSED_DATA_DIR, "experiments.db")

# Налаштування телеметрії дрона (назви колонок, які надсилає Юрій)
TELEMETRY_COLUMNS = [
    "dt", "ax_raw", "ay_raw", "az_raw", 
    "gx_raw", "gy_raw", "gz_raw", 
    "ax_g", "ay_g", "az_g", 
    "gyroX_dps", "gyroY_dps", "gyroZ_dps", 
    "roll_deg", "pitch_deg", "yaw_deg"
]

# Створюємо папки, якщо їх ще немає фізично на диску
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)