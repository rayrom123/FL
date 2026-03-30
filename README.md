# Custom Federated Learning System for IDS

A lightweight, modular Federated Learning (FL) simulation system designed for Intrusion Detection Systems (IDS) using the CIC-IoT23 dataset. This system is built from scratch using PyTorch and does not require external FL frameworks like Flower.

## 🚀 Features

- **Custom FL Simulation**: Complete control over round orchestration, aggregation, and client management.
- **Advanced Strategies**: Supports `FedAvg` (standard averaging) and `FedProx` (handles non-IID data with a proximal term).
- **Multiprocessing**: Parallel local training on multiple CPU processes for high-performance simulation.
- **Robust Feature Engineering**: Automated cleaning, data mapping, and feature selection using Random Forest importance.
- **Non-IID Partitioning**: Dirichlet distribution-based data splitting to simulate realistic heterogeneous client distributions.
- **Comprehensive Evaluation**: Track Global Accuracy, Loss, Precision, Recall, and F1-score across rounds.
- **Automatic Visualization**: Generates training plots (`metrics_plot.png`) and data distribution charts (`distribution_plot.png`).

## 📁 Project Structure

```text
d:/FL/
├── core/
│   ├── client/
│   │   └── client_app.py        # Local training logic & FedProx support
│   ├── data_split/
│   │   └── partitioned_data.py # Data preprocessing & partitioning pipeline
│   ├── dataset/
│   │   └── cic-iot23.csv       # Dataset (not included in repo)
│   ├── model/
│   │   └── model.py             # IDS-MLP architecture
│   ├── module/                  # Core FL modules
│   │   ├── aggregators.py       # Weight aggregation functions
│   │   ├── evaluators.py        # Detailed evaluation metrics
│   │   └── strategies.py        # FedAvg, FedProx strategies
│   └── server/
│       └── fl_simulation.py     # Main simulation orchestrator
├── results/                     # Logs, models, and plots (auto-generated)
└── README.md
```

## 🛠️ Installation

Ensure you have Python 3.8+ and the following libraries installed:

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

## 📋 Usage

### 1. Data Preparation
First, run the data prep pipeline to clean the dataset, select features, and partition data for clients.

```bash
python core/data_split/partitioned_data.py
```
- Outputs: `client_X_data.pt`, `global_test_data.pt`, and `distribution_plot.png`.

### 2. Run FL Simulation
Startup the central server to coordinate client training.

```bash
python core/server/fl_simulation.py
```
- Configuration: You can adjust `NUM_ROUNDS`, `NUM_CLIENTS`, and `FED_ALGO` (`FedAvg` or `FedProx`) inside `fl_simulation.py`.

## 📈 Monitoring
All training logs and metrics are saved in the `results/` folder, organized by timestamp:
- `log.txt`: Detailed training logs.
- `metrics_plot.png`: Global accuracy and loss curves.
- `model.pth`: Final aggregated global model weights.

## ⚖️ License
This project is for educational and research purposes.
