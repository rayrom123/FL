import torch
import torch.nn as nn
import copy
import os
import sys

# Thêm các folder con vào sys.path để có thể import chéo
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.dirname(current_dir)
sys.path.append(core_dir)
sys.path.append(os.path.join(core_dir, "model"))
sys.path.append(os.path.join(core_dir, "client"))

from model import IDS_MLP
from client_app import LocalClient
from dataset_preprocessor import load_and_preprocess_ciciot
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor

# --- CẤU HÌNH ---
NUM_CLIENTS = 5
NUM_ROUNDS = 50  # Giảm xuống 30 rounds như yêu cầu
LOCAL_EPOCHS = 1 # Giảm xuống 1 epoch để train nhanh hơn
BATCH_SIZE = 128 # Tăng Batch Size lên 128
LEARNING_RATE = 0.001
DEVICE = torch.device("cpu") # Trên i7-1255U dùng CPU có Multiprocessing sẽ nhanh hơn GPU nếu tập dữ liệu không quá lớn
GLOBAL_DATA_FILE = "d:/FL/core/dataset/cic-iot23.csv"

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

# ... (các hàm aggregation và evaluate giữ nguyên) ...

def fedprox_aggregation(client_weights, client_samples):
    """
    FedProx uses the same aggregation as FedAvg on the server side.
    The difference is in the client-side loss function.
    """
    return federated_averaging(client_weights, client_samples)

def federated_averaging(client_weights, client_samples):
    """
    Standard FedAvg implementation.
    Weighted average of client parameters.
    """
    total_samples = sum(client_samples)
    global_weights = copy.deepcopy(client_weights[0])
    
    # Initialize global weights to zero
    for key in global_weights.keys():
        global_weights[key] = torch.zeros_like(global_weights[key])
        
    # Aggregate (nen tach module ra, submodule ra de xu ly tung phan rieng biet)
    for i in range(len(client_weights)):
        weight = client_samples[i] / total_samples
        for key in global_weights.keys():
            if torch.is_floating_point(client_weights[i][key]):
                global_weights[key] += client_weights[i][key] * weight
            else:
                if i == 0:
                    global_weights[key] = client_weights[i][key]
                else:
                    global_weights[key] = torch.max(global_weights[key], client_weights[i][key])
            
    return global_weights

def evaluate_global_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
    return avg_loss, accuracy

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

def get_next_run_dir(base_dir="d:/FL/results"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    i = 1
    while os.path.exists(os.path.join(base_dir, f"result_{i}")):
        i += 1
    run_dir = os.path.join(base_dir, f"result_{i}")
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
    
        # 1. Prepare Global Test Set
        print("Loading global test data...")
        X, y = load_and_preprocess_ciciot(GLOBAL_DATA_FILE)
        test_indices = torch.randperm(len(X))[:10000] # Giảm nhẹ số lượng test để nhanh hơn nữa
        test_data = TensorDataset(torch.from_numpy(X[test_indices]), torch.from_numpy(y[test_indices]))
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

        # 2. Initialize Global Model
        global_model = IDS_MLP().to(DEVICE)
        global_weights = global_model.state_dict()
        
        # 3. Prepare Client Data Paths
        data_dir = os.path.join(core_dir, "data_split")
        client_tasks = []
        for i in range(NUM_CLIENTS):
            d_path = os.path.join(data_dir, f"client_{i}_data.pt")
            client_tasks.append((i, d_path))

        # 4. Federated Training Loop
        print(f"\n--- BẮT ĐẦU TRAINING FEDERATED (Algo: {FED_ALGO}) ---")
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
                
            # Global Aggregation
            if FED_ALGO == 'FedProx':
                global_weights = fedprox_aggregation(round_weights, round_samples)
            else:
                global_weights = federated_averaging(round_weights, round_samples)
                
            global_model.load_state_dict(global_weights)
            
            global_loss, global_acc = evaluate_global_model(global_model, test_loader, DEVICE)
            print(f"Global Model Round {round_idx + 1} | Accuracy: {global_acc:.2f}% | Loss: {global_loss:.4f}")
            
            history_acc.append(global_acc)
            history_loss.append(global_loss)

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
