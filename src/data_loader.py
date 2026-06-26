import os
import pandas as pd
from src.config import RAW_DATA_DIR, TELEMETRY_COLUMNS

def load_telemetry_csv(filename):
    """Завантажує окремий CSV файл випробування у формат DataFrame."""
    file_path = os.path.join(RAW_DATA_DIR, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {filename} не знайдено в папці {RAW_DATA_DIR}")
        
    # Зчитуємо CSV. Якщо заголовки відсутні, ми передаємо їх з конфігу через names
    df = pd.read_csv(file_path)
    return df