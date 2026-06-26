import os
import sys
import time
import json
import urllib.request

# 1. Знаходимо корінь проєкту 
current_file_dir = os.path.dirname(os.path.abspath(__file__)) # це папка src
project_root = os.path.abspath(os.path.join(current_file_dir, '..')) # це папка AI_proj

# 2. Примусово додаємо корінь проєкту в самий початок шляхів пошуку Python
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"🚀 Системні шляхи налаштовано! Корінь проєкту: {project_root}")

# 3. імпортуємо модулі через повний шлях 'src.', щоб усередині них нічого не ламалося
from src.database import init_db, save_experiment_result
from src.data_loader import load_telemetry_csv
from src.metrics import calculate_fitness_v1
from src.ga_optimizer import GeneticPIDOptimizer

def run_pipeline(mode="csv", csv_filename="test_run_01.csv"):
    """
    Головний запуск системи:
    mode="csv" -> аналізує готовий файл з папки data/raw
    mode="stream" -> підключається наживо до дрона Юрія по Wi-Fi
    """
    print("\n" + "="*50)
    print("СИСТЕМА ОПТИМІЗАЦІЇ ДРОНА (MAIN PIPELINE) ЗАПУЩЕНА")
    print("="*50)
    
    # 1. Ініціалізуємо базу даних
    init_db()
    
    # 2. Створюємо ШІ-оптимізатор та генеруємо поточну популяцію
    optimizer = GeneticPIDOptimizer(population_size=5)
    initial_population = optimizer.init_population()
    
    # Беремо першого кандидата, якого ми зараз тестуємо
    current_candidate = initial_population[0]
    print(f"\nТестуємо кандидата №1 з параметрами PID:")
    print(f"   Kp: {current_candidate['kp']} | Ki: {current_candidate['ki']} | Kd: {current_candidate['kd']}\n")
    
    real_angles = []
    
    # 3. ЗБІР ДАНИХ (Залежно від режиму)
    if mode == "stream":
        STREAM_URL = "http://192.168.4.1/stream"
        print(f"📡 Режим СТРІМУ: Підключаюся до {STREAM_URL}...")
        try:
            stream = urllib.request.urlopen(STREAM_URL, timeout=5)
            start_time = time.time()
            while time.time() - start_time < 5:  # Збираємо дані 5 секунд
                line = stream.readline().decode('utf-8').strip()
                if line.startswith("data:"):
                    json_data = json.loads(line.replace("data:", "").strip())
                    real_angles.append(json_data["imu"]["pitch_deg"])
            stream.close()
            print(f"✅ Дані зі стріму зібрано ({len(real_angles)} точок)")
        except Exception as e:
            print(f"❌ Помилка підключення до стріму: {e}")
            print("Перемикаюся на резервний CSV файл...")
            mode = "csv"
            
    if mode == "csv":
        print(f"Режим ФАЙЛУ: Зчитую історичний лог {csv_filename}...")
        try:
            df = load_telemetry_csv(csv_filename)
            real_angles = df['pitch_deg'].tolist()
            print(f"✅ Файл успішно зчитано ({len(real_angles)} точок)")
        except Exception as e:
            print(f"❌ Помилка зчитування файлу: {e}")
            return

    # 4. МАТЕМАТИЧНА ОЦІНКА ПОЛЬОТУ
    if not real_angles:
        print("Немає даних для аналізу кутів!")
        return
        
    score = calculate_fitness_v1(real_angles)
    print(f"Розрахована оцінка стабільності: {score:.2f} балів із 100")
    
    # 5. ЗБЕРЕЖЕННЯ В БАЗУ ДАНИХ
    save_experiment_result(
        kp=current_candidate['kp'],
        ki=current_candidate['ki'],
        kd=current_candidate['kd'],
        fitness_version="v1.0",
        fitness_score=score,
        status="success" if score > 40 else "failed",
        csv_filename=csv_filename if mode == "csv" else "live_stream.csv"
    )
    print("Результат успішно зафіксовано в базі даних experiments.db!")
    print("="*50 + "\n")

if __name__ == "__main__":
    # За замовчуванням запускаємо аналіз нашого файлу
    run_pipeline(mode="csv", csv_filename="test_run_01.csv")