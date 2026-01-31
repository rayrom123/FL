import torch
import torch.nn as nn
import copy
import os
import sys

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.dirname(current_dir)
sys.path.append(core_dir)
sys.path.append(os.path.join(core_dir, "model"))
sys.path.append(os.path.join(core_dir, "client"))
sys.path.append(os.path.join(core_dir, "federated"))

# Project imports
from model import IDS_MLP
from client_app import LocalClient
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Federated Learning modules
from module import create_strategy, evaluate_model, format_metrics

from concurrent.futures import ProcessPoolExecutor

# --- CẤU HÌNH ---
NUM_CLIENTS = 5
NUM_ROUNDS = 50  
LOCAL_EPOCHS = 1 # Giảm xuống 1 epoch để train nhanh hơn
BATCH_SIZE = 128 # Tăng Batch Size lên 128
LEARNING_RATE = 0.001
DEVICE = torch.device("cpu") # Trên i7-1255U dùng CPU có Multiprocessing sẽ nhanh hơn GPU nếu tập dữ liệu không quá lớn
GLOBAL_TEST_FILE = "global_test_data.pt"  # Pre-split test set (30%)

# --- FL ALGORITHM CONFIG ---
# FED_ALGO = 'FedProx' 
FED_ALGO = 'FedAvg' 
PROXIMAL_MU = 0.01 if FED_ALGO == 'FedProx' else 0.0

def client_train_worker(client_id, data_path, global_weights, current_lr):
    """
    Worker function chạy trên từng Process riêng biệt.
    """
    # Khởi tạo client mới trong process này để tránh xung đột bộ nhớ
    client = LocalClient(client_id, torch.device("cpu"), data_path=data_path, batch_size=BATCH_SIZE)
    client.set_parameters(global_weights)
    
    weights, samples = client.train(
        global_params=global_weights, 
        epochs=LOCAL_EPOCHS, 
        lr=current_lr,
        proximal_mu=PROXIMAL_MU
    )
    
    loss, acc = client.evaluate()
    return client_id, weights, samples, loss, acc

# ========================================
# Aggregation and Evaluation
# Now using federated modules!
# ========================================

def save_training_plots(run_dir, accuracies, losses):
    """
    Vẽ biểu đồ Accuracy và Loss sau khi train xong.
    """
    rounds = range(1, len(accuracies) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(rounds, accuracies, 'b-o', label='Global Accuracy')
    plt.title('Global Model Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(rounds, losses, 'r-o', label='Global Loss')
    plt.title('Global Model Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(run_dir, "metrics_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"-> Đã lưu biểu đồ tại: {plot_path}")

import sys

class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Quan trọng: Phải flush để ghi vào file ngay lập tức

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        if self.log:
            self.log.close()

def get_next_run_dir(base_dir="c:/FederatedLearning/core/results"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Tạo tên folder theo ngày giờ: DD-MM-YYYY_HH-MM-SS
    from datetime import datetime
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    run_dir = os.path.join(base_dir, timestamp)
    
    # Nếu trùng tên (chạy trong cùng 1 giây), thêm suffix   
    if os.path.exists(run_dir):
        i = 1
        while os.path.exists(f"{run_dir}_{i}"):
            i += 1
        run_dir = f"{run_dir}_{i}"
    
    os.makedirs(run_dir)
    return run_dir

def main():
    # 0. Setup Result Directory and Logging
    run_dir = get_next_run_dir()
    log_file = os.path.join(run_dir, "log.txt")
    
    logger = Logger(log_file)
    original_stdout = sys.stdout
    sys.stdout = logger
    
    try:
        print(f"--- TRAINING SESSION: {os.path.basename(run_dir)} ---")
        print(f"Multiprocessing enabled for performance optimization.")
    
        # 1. Load Pre-split Global Test Set (tránh Data Leakage)
        print("Loading pre-split global test data...")
        test_data_path = os.path.join(core_dir, "data_split", GLOBAL_TEST_FILE)
        X_test, y_test = torch.load(test_data_path)
        test_data = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
        print(f"→ Loaded {len(X_test)} test samples (30% of total data)")
        print(f"→ Features: {X_test.shape[1]}")

        # 2. Initialize Global Model
        global_model = IDS_MLP().to(DEVICE)
        global_weights = global_model.state_dict()
        
        # 3. Prepare Client Data Paths
        data_dir = os.path.join(core_dir, "data_split")
        client_tasks = []
        for i in range(NUM_CLIENTS):
            d_path = os.path.join(data_dir, f"client_{i}_data.pt")
            client_tasks.append((i, d_path))

        # 4. Initialize FL Strategy
        print(f"\n--- BẮT ĐẦU TRAINING FEDERATED (Algo: {FED_ALGO}) ---")
        if FED_ALGO == 'FedProx':
            strategy = create_strategy('FedProx', mu=PROXIMAL_MU)
        else:
            strategy = create_strategy('FedAvg')
        print(f"→ Strategy: {strategy.name}")
        
        current_lr = LEARNING_RATE
        
        history_acc = []
        history_loss = []
        
        for round_idx in range(NUM_ROUNDS):
            print(f"\nRound {round_idx + 1}/{NUM_ROUNDS} | LR: {current_lr:.6f}")
            
            if (round_idx + 1) % 10 == 0:
                current_lr *= 0.5
                print(f"  [Info] Learning Rate decayed to {current_lr:.6f}")

            round_weights = []
            round_samples = []
            
            # CHẠY SONG SONG CÁC CLIENTS
            print(f"  Training {NUM_CLIENTS} clients in parallel...")
            with ProcessPoolExecutor(max_workers=NUM_CLIENTS) as executor:
                futures = [executor.submit(client_train_worker, cid, dp, global_weights, current_lr) for cid, dp in client_tasks]
                
                for future in futures:
                    cid, weights, samples, loss, acc = future.result()
                    print(f"  Client {cid} finished | Acc: {acc:.2f}% | Loss: {loss:.4f}")
                    round_weights.append(weights)
                    round_samples.append(samples)
            
            # Global Aggregation using strategy
            global_weights = strategy.aggregate(round_weights, round_samples)
            global_model.load_state_dict(global_weights)
            
            # Evaluate with detailed metrics (verbose on last round)
            verbose = (round_idx == NUM_ROUNDS - 1)
            metrics = evaluate_model(global_model, test_loader, DEVICE, verbose=verbose)
            print(f"Global Model Round {round_idx + 1} | {format_metrics(metrics)}")
            
            history_acc.append(metrics['accuracy'])
            history_loss.append(metrics['loss'])

        # 5. Save Final Model to the run directory
        model_path = os.path.join(run_dir, "model.pth")
        torch.save(global_model.state_dict(), model_path)
        
        # 6. Vẽ biểu đồ
        save_training_plots(run_dir, history_acc, history_loss)
        
        print(f"\n--- Simulation Complete ---")
    finally:
        sys.stdout = original_stdout
        logger.close()

if __name__ == "__main__":
    main()


