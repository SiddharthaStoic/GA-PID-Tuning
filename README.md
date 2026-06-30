# 🚁 AI-Driven Genetic Algorithm Framework for Autonomous PID Tuning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)

An end-to-end AI optimization framework for autonomous PID controller tuning using Genetic Algorithms (GA), telemetry processing, system identification, multi-objective fitness evaluation, and experiment tracking.

---

## 🚀 Project Overview

Manual PID tuning is a time-consuming trial-and-error process. This project automates PID tuning by combining a Digital Twin simulation with a Genetic Algorithm to search for PID gains that minimize tracking error, overshoot, settling time, and control effort while maintaining flight stability.

---

## ✨ Features

- Modular AI architecture
- Telemetry loading and preprocessing
- Signal filtering
- System identification
- Feature extraction
- Multi-objective fitness evaluation
- Genetic Algorithm optimization
- SQLite experiment tracking
- Scientific visualizations
- Research notebooks

---

## 📂 Repository Structure

```text
AI_GA_PID_Tuning/
├── data/
│   ├── raw/
│   ├── processed/
│   └── experiments.db
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── config.py
│   ├── telemetry.py
│   ├── processing.py
│   ├── simulator.py
│   ├── features.py
│   ├── fitness.py
│   ├── ga_optimizer.py
│   ├── database.py
│   └── main.py
└── README.md
```

---

## 🧠 AI Pipeline

1. Load telemetry
2. Preprocess signals
3. Identify system dynamics
4. Extract performance features
5. Evaluate fitness
6. Optimize PID gains with GA
7. Store experiment history
8. Visualize results

---

## 📊 Results

### Optimization Progress

![Optimization Progress](reports/figures/optimization_progress.png)

### Controller Responses

![Controller Responses](reports/figures/controller_responses.png)

### Fitness Comparison

![Fitness Comparison](reports/figures/fitness_comparison.png)

### Population Diversity

![Population Diversity](reports/figures/population_diversity.png)

### System Identification

![Measured vs Simulated](reports/figures/system_identification.png)

> Rename your image files in `reports/figures/` to match the names above, or update the image paths accordingly.

---

## 💻 Technologies

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- SQLite
- Jupyter Notebook

---

## ▶️ Running

```bash
pip install numpy pandas scipy matplotlib
python -m src.main
```

---

## 🔮 Future Work

- Neural-network system identification
- Particle Swarm Optimization
- Bayesian Optimization
- Multi-axis PID tuning
- Hardware-in-the-loop testing
- Real-time telemetry streaming

---