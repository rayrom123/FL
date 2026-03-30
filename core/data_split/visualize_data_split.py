import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Path setup
data_dir = r"d:\FL\core\data_split"
centralized_dir = os.path.join(data_dir, "centralized_data")
federated_dir = os.path.join(data_dir, "federated_data")
tasks = [1, 2, 3, 4, 5, 6]
num_clients = 5

# Label grouping for visualization (matches partitioned_data.py)
TASK_CONFIG = {
    1: [0, 1, 24, 25, 26, 27],
    2: [2, 13, 11, 12, 29, 30],
    3: [3, 14, 16, 17, 18, 19],
    4: [4, 15, 20, 21, 22, 23],
    5: [5, 28, 8, 9, 10],
    6: [6, 7, 31, 32, 33]
}

def plot_task_comparison():
    """Vẽ biểu đồ cột so sánh Train/Test cho từng Task (Centralized)."""
    tr_totals, te_totals = [], []
    for t in tasks:
        tr_path = os.path.join(centralized_dir, f"task_{t}_train.pt")
        te_path = os.path.join(centralized_dir, f"task_{t}_test.pt")
        tr_count = len(torch.load(tr_path)[1]) if os.path.exists(tr_path) else 0
        te_count = len(torch.load(te_path)[1]) if os.path.exists(te_path) else 0
        tr_totals.append(tr_count)
        te_totals.append(te_count)
        
    x = np.arange(len(tasks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, tr_totals, width, label='Train', color='skyblue', edgecolor='navy')
    ax.bar(x + width/2, te_totals, width, label='Test', color='salmon', edgecolor='darkred')
    ax.set_ylabel('Samples')
    ax.set_title('Centralized Tasks: Train vs Test Counts', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Task {t}" for t in tasks])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "task_train_test_comparison.png"), dpi=200)
    print("✓ Đã lưu task_train_test_comparison.png")
    plt.close()

def plot_task_bubble():
    """Vẽ bubble chart cho phân phối nhãn trong các Tasks (Centralized Train)."""
    x_data, y_data, counts = [], [], []
    for t in tasks:
        path = os.path.join(centralized_dir, f"task_{t}_train.pt")
        if os.path.exists(path):
            _, y = torch.load(path)
            unique, c_vals = np.unique(y.numpy(), return_counts=True)
            for u, c in zip(unique, c_vals):
                x_data.append(t)
                y_data.append(int(u))
                counts.append(int(c))
                
    fig, ax = plt.subplots(figsize=(12, 10))
    max_c = max(counts) if counts else 1
    sizes = [(v / max_c) * 2000 + 100 for v in counts]
    cmap = matplotlib.colormaps.get_cmap('tab20')
    colors = [cmap(val % 20 / 20) for val in y_data]
    
    ax.scatter(x_data, y_data, s=sizes, c=colors, alpha=0.6, edgecolors='black')
    ax.set_xlabel('Task ID')
    ax.set_ylabel('Label ID')
    ax.set_title('Task-Label Distribution (Centralized Train)', fontsize=14, fontweight='bold')
    ax.set_xticks(tasks)
    ax.set_yticks(range(34))
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "task_label_distribution.png"), dpi=200)
    print("✓ Đã lưu task_label_distribution.png")
    plt.close()

def plot_federated_bubble_grid():
    """
    Vẽ bubble chart cho phân phối nhãn trên các Clients, chia theo Task (GRID 3x2).
    Thể hiện rõ tính chất Task-specific Federated Learning.
    """
    # Màu sắc cố định cho các nhãn
    cmap = matplotlib.colormaps.get_cmap('tab20b')
    label_colors = {lbl: cmap(lbl % 20 / 20) for lbl in range(34)}
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()
    
    print("--- Đang tạo federated_task_client_distribution.png (Grid view) ---")
    
    for idx, t in enumerate(tasks):
        ax = axes[idx]
        task_labels = TASK_CONFIG[t]
        
        x_data, y_data, counts, colors = [], [], [], []
        
        # Đọc dữ liệu của từng client cho task này
        for i in range(num_clients):
            path = os.path.join(federated_dir, f"client{i}_task{t}.pt")
            if os.path.exists(path):
                _, y = torch.load(path)
                unique, c_vals = np.unique(y.numpy(), return_counts=True)
                for u, c in zip(unique, c_vals):
                    x_data.append(i)
                    y_data.append(int(u))
                    counts.append(int(c))
                    colors.append(label_colors[int(u)])
        
        if counts:
            max_c = max(counts)
            sizes = [(v / max_c) * 1500 + 100 for v in counts]
            ax.scatter(x_data, y_data, s=sizes, c=colors, alpha=0.6, edgecolors='black', linewidths=0.8)
            
        ax.set_title(f"Task {t} (Labels: {task_labels})", fontsize=13, fontweight='bold')
        ax.set_xticks(range(num_clients))
        ax.set_xticklabels([f"Client {i}" for i in range(num_clients)])
        ax.set_yticks(task_labels)
        ax.grid(True, alpha=0.3, linestyle='--')
        
    plt.suptitle("Federated Continual Learning: Task-Client Label Distribution", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    save_path = os.path.join(data_dir, "federated_task_client_distribution.png")
    plt.savefig(save_path, dpi=200)
    print("✓ Đã lưu federated_task_client_distribution.png (Dạnh lưới 6 tasks)")
    plt.close()

if __name__ == "__main__":
    print("--- Đang tạo các biểu đồ phân phối mới ---")
    plot_task_comparison()
    plot_task_bubble()
    plot_federated_bubble_grid()
    print("--- Hoàn tất ---")
