import sqlite3
from src.config import DB_PATH

def init_db():
    """Створює таблицю для збереження результатів тестів, якщо вона не існує."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Створюємо таблицю експериментів
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            kp REAL,                  -- Пропорційний коефіцієнт PID
            ki REAL,                  -- Інтегруючий коефіцієнт PID
            kd REAL,                  -- Диференціюючий коефіцієнт PID
            fitness_version TEXT,     -- Версія нашої fitness-функції (напр., 'v1.0')
            fitness_score REAL,       -- Підсумкова оцінка якості польоту
            status TEXT,              -- Статус: 'success', 'failed' (дрон упав) тощо
            csv_filename TEXT         -- Посилання на сирий CSV файл логу цього тесту
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Базу даних успішно ініціалізовано за шляхом: {DB_PATH}")

def save_experiment_result(kp, ki, kd, fitness_version, fitness_score, status, csv_filename):
    """Функція для запису результату нового тесту в базу."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO test_results (kp, ki, kd, fitness_version, fitness_score, status, csv_filename)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (kp, ki, kd, fitness_version, fitness_score, status, csv_filename))
    
    conn.commit()
    conn.close()
    print("Результат експерименту успешно збережено в SQL!")