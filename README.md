# Custom Federated Learning System for CIC-IoT 2023

This repository contains a custom-built Federated Learning (FL) system designed for the CIC-IoT 2023 dataset. It implements core FL concepts like **FedAvg** and **FedProx** from scratch using PyTorch, optimized for high-performance execution on multi-core CPUs.

## 🚀 Key Features
- **Custom FL Framework**: Built without external libraries like Flower or PySyft.
- **FedProx Support**: Handles Non-IID data by adding a proximal term to the local loss.
- **Multiprocessing**: Parallelizes client training to drastically reduce execution time.
- **Automated Visualization**: Generates Accuracy and Loss plots for every training session.
- **Result Archiving**: Automatically organizes logs, models, and plots into versioned folders.

## 📂 Project Structure
```text
core/
├── client/
│   └── client_app.py        # Local training logic
├── data_split/
│   └── partitioned_data.py  # Non-IID data partitioning script
├── dataset/
│   └── (Put cic-iot23.csv here)
├── model/
│   └── model.py             # IDS MLP Architecture
├── server/
│   ├── fl_simulation.py     # Main orchestrator (Server)
│   └── generate_plots.py    # Utility to re-generate plots
└── dataset_preprocessor.py  # Shared data cleaning logic
```

## 🛠️ How to Run

### 1. Requirements
Install the necessary Python packages:
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

### 2. Data Preparation
1. Download the **CIC-IoT 2023** dataset and place the `cic-iot23.csv` file in `core/dataset/`.
2. Run the partitioning script to create Non-IID data for clients:
   ```powershell
   python core/data_split/partitioned_data.py
   ```
   *This will generate `.pt` files in `core/data_split/`.*

### 3. Start Training
Run the main simulation script:
```powershell
python core/server/fl_simulation.py
```

## 📊 Results and Logging
Every training session creates a new folder in `results/result_N/` containing:
- `log.txt`: Detailed training logs (Accuracy/Loss per round).
- `metrics_plot.png`: Visual chart of the training progress.
- `model.pth`: The final aggregated global model weights.

## ⚙️ Configuration
You can adjust the following parameters in `core/server/fl_simulation.py`:
- `NUM_ROUNDS`: Total number of communication rounds.
- `FED_ALGO`: Switch between `'FedAvg'` and `'FedProx'`.
- `BATCH_SIZE` & `LEARNING_RATE`: Standard training hyperparameters.
- `PROXIMAL_MU`: The $\mu$ parameter for FedProx.
