import random

class GeneticPIDOptimizer:
    def __init__(self, population_size=10):
        self.population_size = population_size
        self.population = []
        
    def generate_random_pid(self):
        """Генерує випадковий набір коефіцієнтів Kp, Ki, Kd у розумних межах."""
        kp = round(random.uniform(0.1, 5.0), 3)
        ki = round(random.uniform(0.0, 1.0), 3)
        kd = round(random.uniform(0.0, 2.0), 3)
        return {"kp": kp, "ki": ki, "kd": kd}
        
    def init_population(self):
        """Створює першу 'популяцію' (набір кандидатів) для першого етапу тестів."""
        self.population = [self.generate_random_pid() for _ in range(self.population_size)]
        return self.population

    def select_best_candidates(self, rated_population, num_best=2):
        """
        Приймає список кандидатів з їхніми оцінками (fitness_score)
        і відбирає найкращих для схрещування.
        Сортуємо за спаданням оцінки (число від 100 до 0).
        """
        # rated_population має бути у форматі: [({"kp": 1.2, ...}, 85.5), ...]
        sorted_pop = sorted(rated_population, key=lambda x: x[1], reverse=True)
        return sorted_pop[:num_best]