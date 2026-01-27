# Custom Federated Learning System for CIC-IoT 2023

This project implements a custom Federated Learning (FL) system using PyTorch, designed to train an Intrusion Detection System (IDS) on the CIC-IoT 2023 dataset without external FL frameworks like Flower.

## System Architecture

The system consists of:
- **`model.py`**: A Multi-Layer Perceptron (MLP) architecture suitable for 45 features and 34 output classes.
- **`dataset_preprocessor.py`**: Shared utility for cleaning and normalizing the CIC-IoT dataset.
- **`partitioned_data.py`**: Splits the global dataset into non-IID partitions for each client using Dirichlet distribution.
- **`client_app.py`**: Implements the `LocalClient` class, which handles local weights, training loop, and evaluation.
- **`fl_simulation.py`**: The central orchestrator (Server) that manages rounds, implements **FedAvg** aggregation, and evaluates the global model.

## How to Run

### 1. Requirements
Ensure you have the following installed:
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

### 2. Prepare Data
First, you need to partition the global `cic-iot23.csv` file into client-specific files:
```powershell
python core/data_split/partitioned_data.py
```
This will generate files like `client_0_data.pt`, `client_1_data.pt`, etc.

### 3. Run Simulation
Start the federated training process:
```powershell
python core/server/fl_simulation.py
```

## Core Logic: FedAvg
The system uses the Federated Averaging (FedAvg) algorithm, where the global model weights are updated as a weighted average of client weights based on their local sample counts:

$$w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_{t+1}^k$$

- $n_k$: Number of samples at client $k$
- $n$: Total number of samples across all selected clients
- $w_{t+1}^k$: Updated weights from client $k$


## Results and Logging
Training results are saved in `d:/FL/results/result_N/`:
- **`log.txt`**: Detailed logs of training performance and global evaluation.
- **`model.pth`**: Final aggregated global model weights.