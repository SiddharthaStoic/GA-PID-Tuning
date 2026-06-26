import numpy as np

def calculate_mae_error(current_angles, target_angle=0.0):
    """
    Рахує середню абсолютну помилку (Mean Absolute Error).
    Чим менше це число, тим стабільніше тримався дрон.
    """
    # Перетворюємо в масив numpy для швидкості розрахунків
    angles = np.array(current_angles)
    # Рахуємо абсолютну різницю між поточним кутом і цільовим (0)
    errors = np.abs(angles - target_angle)
    # Повертаємо середнє значення
    return float(np.mean(errors))

def calculate_fitness_v1(current_angles, target_angle=0.0):
    """
    Fitness-функція Версії 1.0.
    Перетворює помилку в 'бали' від 0 до 100.
    100 — ідеальний політ без коливань.
    0 — дрон сильно бовтало.
    """
    if len(current_angles) == 0:
        return 0.0
        
    mae = calculate_mae_error(current_angles, target_angle)
    max_error = float(np.max(np.abs(np.array(current_angles) - target_angle)))
    
    # Формула оцінки: базово даємо 100 балів і віднімаємо штраф за помилки.
    # Вагові коефіцієнти : середня помилка важливіша, тому множимо її на 5, 
    # а максимальний спалах — на 1.
    score = 100.0 - (mae * 5.0) - (max_error * 1.0)
    
    # Оцінка не може бути меншою за 0
    return max(0.0, score)