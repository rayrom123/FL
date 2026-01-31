import os
import sys
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.dirname(current_dir)
if core_dir not in sys.path:
    sys.path.append(core_dir)

# CẤU HÌNH
FILE_PATH = r"C:\FederatedLearning\core\dataset\cic-iot23.csv"
NUM_CLIENTS = 5
ALPHA = 0.5  # Độ lệch Non-IID
TRAIN_RATIO = 0.7  # 70% train, 30% test
FEATURE_SELECTION_THRESHOLD = 0.001  # Ngưỡng để loại bỏ features không quan trọng
RANDOM_SEED = 42  # Fix seed để kết quả reproducible

# FIX RANDOM SEED cho tất cả thư viện
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def load_and_clean_data(file_path):
    """
    Load và clean dữ liệu CIC-IoT23.
    Trả về X (features) và y (labels).
    """
    print(f"--- Đang load và clean dữ liệu từ: {file_path} ---")
    
    # 1. Load dữ liệu
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Đã load {len(df)} dòng, {len(df.columns)} cột")
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return None, None

    # 2. Xóa cột rác (nếu có)
    if 'Number' in df.columns:
        print("→ Đã xóa cột 'Number'")
        df.drop(columns=['Number'], inplace=True)
    
    # 3. Xử lý Inf/NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"→ Phát hiện {missing_count} giá trị lỗi (NaN/Inf). Đang xóa dòng...")
        df.dropna(inplace=True)

    # 4. Mapping nhãn
    full_label_mapping = {
        'BenignTraffic': 0,
        'DDoS-ICMP_Flood': 1, 'DDoS-UDP_Flood': 2, 'DDoS-TCP_Flood': 3,
        'DDoS-PSHACK_Flood': 4, 'DDoS-SYN_Flood': 5, 'DDoS-RSTFINFlood': 6,
        'DDoS-SynonymousIP_Flood': 7, 'DDoS-ICMP_Fragmentation': 8,
        'DDoS-UDP_Fragmentation': 9, 'DDoS-ACK_Fragmentation': 10,
        'DDoS-HTTP_Flood': 11, 'DDoS-SlowLoris': 12,
        'DoS-UDP_Flood': 13, 'DoS-TCP_Flood': 14, 'DoS-SYN_Flood': 15, 'DoS-HTTP_Flood': 16,
        'Recon-HostDiscovery': 17, 'Recon-OSScan': 18, 'Recon-PortScan': 19,
        'Recon-PingSweep': 20, 'VulnerabilityScan': 21,
        'MITM-ArpSpoofing': 22, 'DNS_Spoofing': 23,
        'DictionaryBruteForce': 24,
        'BrowserHijacking': 25, 'XSS': 26, 'Uploading_Attack': 27,
        'SqlInjection': 28, 'CommandInjection': 29, 'Backdoor_Malware': 30,
        'Mirai-greeth_flood': 31, 'Mirai-udpplain': 32, 'Mirai-greip_flood': 33
    }
    
    df['label_code'] = df['label'].map(full_label_mapping)
    
    if df['label_code'].isnull().any():
        unknowns = df[df['label_code'].isnull()]['label'].unique()
        print(f"!!! CẢNH BÁO: Bỏ qua các nhãn lạ: {unknowns}")
        df = df.dropna(subset=['label_code'])

    y = df['label_code'].values.astype(np.int64)
    
    # 5. Tách features
    X_raw = df.drop(columns=['label', 'label_code'])
    print(f"→ Số lượng Features ban đầu: {X_raw.shape[1]}")
    
    return X_raw, y

def feature_engineering(X_raw, y, threshold=0.001):
    """
    Feature engineering: loại bỏ các features không đóng góp.
    Sử dụng Random Forest để đánh giá feature importance.
    """
    print(f"\n--- BẮT ĐẦU FEATURE ENGINEERING ---")
    print(f"→ Ngưỡng importance: {threshold}")
    
    # 1. Scale features trước (để RF hoạt động tốt hơn)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)
    
    # 2. Train Random Forest nhanh để đánh giá importance
    print("→ Đang train Random Forest để đánh giá feature importance...")
    # Lấy mẫu nhỏ để train nhanh (10% hoặc max 50k samples)
    sample_size = min(50000, int(len(X_scaled) * 0.1))
    indices = np.random.choice(len(X_scaled), sample_size, replace=False)
    
    rf = RandomForestClassifier(
        n_estimators=50,  # Số cây ít để train nhanh
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_scaled[indices], y[indices])
    
    # 3. Lấy feature importance
    importances = rf.feature_importances_
    feature_names = X_raw.columns.tolist()
    
    # Sắp xếp theo importance
    indices_sorted = np.argsort(importances)[::-1]
    
    print("\n→ Top 10 features quan trọng nhất:")
    for i in range(min(10, len(indices_sorted))):
        idx = indices_sorted[i]
        print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # 4. Chọn features có importance > threshold
    selected_indices = importances > threshold
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_indices[i]]
    removed_features = [feature_names[i] for i in range(len(feature_names)) if not selected_indices[i]]
    
    print(f"\n→ Số features GIỮ LẠI: {len(selected_features)}")
    print(f"→ Số features BỊ LOẠI: {len(removed_features)}")
    
    if len(removed_features) > 0:
        print(f"\n→ Các features bị loại bỏ:")
        for feat in removed_features:
            idx = feature_names.index(feat)
            print(f"   - {feat}: {importances[idx]:.6f}")
    
    # 5. Lọc X_raw và scale lại
    X_filtered = X_raw[selected_features]
    scaler_final = MinMaxScaler()
    X_final = scaler_final.fit_transform(X_filtered).astype(np.float32)
    
    print(f"\n→ Shape cuối cùng: {X_final.shape}")
    
    return X_final, y, selected_features, scaler_final

def split_train_test(X, y, train_ratio=0.7):
    """
    Chia dữ liệu thành train và test theo tỷ lệ.
    Sử dụng stratified split để đảm bảo phân phối labels cân bằng.
    """
    print(f"\n--- CHIA DỮ LIỆU TRAIN/TEST ({int(train_ratio*100)}/{int((1-train_ratio)*100)}) ---")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_ratio,
        stratify=y,  # Đảm bảo phân phối labels cân bằng
        random_state=42
    )
    
    print(f"→ Train set: {X_train.shape[0]} samples")
    print(f"→ Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def partition_data_non_iid(X_train, y_train, num_clients=5, alpha=0.5):
    """
    Chia dữ liệu train thành non-IID partitions cho các clients.
    Sử dụng Dirichlet distribution.
    """
    n_samples = y_train.shape[0]
    n_classes = len(np.unique(y_train))
    
    # Tạo Dictionary chứa index của từng class
    class_indices = [np.argwhere(y_train == k).flatten() for k in range(n_classes)]
    
    # Khởi tạo list chứa index cho từng client
    client_indices = [[] for _ in range(num_clients)]
    
    print(f"\n--- CHIA NON-IID CHO {num_clients} CLIENTS (Alpha={alpha}) ---")
    
    # Thuật toán Dirichlet Partitioning
    for k in range(n_classes):
        idx_k = class_indices[k]
        np.random.shuffle(idx_k)
        
        # Tạo phân phối xác suất cho class k trên các clients
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Cân đối lại proportions
        proportions = np.array([p * (len(idx_k) < n_samples / num_clients) for p in proportions])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        idx_split = np.split(idx_k, proportions)
        
        # Gán index vào từng client
        for i in range(num_clients):
            client_indices[i] += idx_split[i].tolist()

    stats = []
    
    for i in range(num_clients):
        indices = client_indices[i]
        np.random.shuffle(indices)  # Trộn lại để tránh theo thứ tự class
        
        X_client = torch.tensor(X_train[indices], dtype=torch.float32)
        y_client = torch.tensor(y_train[indices], dtype=torch.long)
        
        save_path = os.path.join(core_dir, "data_split", f"client_{i}_data.pt")
        torch.save((X_client, y_client), save_path)
        
        # Thống kê label
        unique, counts = np.unique(y_train[indices], return_counts=True)
        stats.append(dict(zip(unique, counts)))
        
        print(f"→ Client {i}: {len(indices)} mẫu. Đã lưu '{os.path.basename(save_path)}'")

    return stats

def save_global_test_set(X_test, y_test):
    """
    Lưu global test set cho server.
    """
    print(f"\n--- LƯU GLOBAL TEST SET ---")
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    save_path = os.path.join(core_dir, "data_split", "global_test_data.pt")
    torch.save((X_test_tensor, y_test_tensor), save_path)
    
    print(f"→ Đã lưu {len(X_test)} mẫu test tại: {os.path.basename(save_path)}")
    print(f"→ Label distribution trong test set:")
    unique, counts = np.unique(y_test, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"   Class {label}: {count} samples")

def visualize_distribution(stats):
    """
    Vẽ biểu đồ 2D dạng bubble chart (chấm tròn) để thấy độ Non-IID của dữ liệu.
    Trục X (ngang): Client ID
    Trục Y (dọc): Class Label
    Kích thước chấm tròn: Tỷ lệ với số lượng mẫu
    """
    classes = set()
    for s in stats:
        classes.update(s.keys())
    classes = sorted(list(classes))
    
    # Tạo figure cho 2D plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Chuẩn bị dữ liệu cho 2D scatter plot
    x_data = []  # Client IDs (ngang)
    y_data = []  # Class labels (dọc)
    sizes = []   # Kích thước chấm tròn
    colors = []
    
    # Tạo colormap để phân biệt các classes
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap('tab20')
    class_colors = {cls: cmap(i / len(classes)) for i, cls in enumerate(classes)}
    
    # Thu thập dữ liệu
    max_count = 0
    data_values = []
    for client_id in range(NUM_CLIENTS):
        for cls in classes:
            count = stats[client_id].get(cls, 0)
            if count > 0:  # Chỉ vẽ những điểm có dữ liệu
                x_data.append(client_id)
                y_data.append(int(cls))
                data_values.append(count)
                colors.append(class_colors[cls])
                max_count = max(max_count, count)
    
    # Tính toán kích thước chấm tròn (tỷ lệ với số lượng mẫu)
    # Kích thước từ 100 đến 2000 pixels cho 2D
    sizes = [(val / max_count) * 1900 + 100 for val in data_values]
    
    # Vẽ 2D scatter plot
    scatter = ax.scatter(x_data, y_data, 
                        s=sizes, 
                        c=colors, 
                        alpha=0.6,
                        edgecolors='black',
                        linewidths=1.5)
    
    # Cài đặt labels và title
    ax.set_xlabel('Client ID (ngang)', fontsize=14, labelpad=12, fontweight='bold')
    ax.set_ylabel('Class Label (dọc)', fontsize=14, labelpad=12, fontweight='bold')
    ax.set_title(f'Data Distribution - Bubble Chart (Alpha={ALPHA})\nKích thước chấm tròn tỷ lệ với số lượng mẫu', 
                 fontsize=16, pad=20, fontweight='bold')
    
    # Cài đặt các ticks
    ax.set_xticks(range(NUM_CLIENTS))
    ax.set_yticks(classes)
    
    # Thêm grid để dễ nhìn
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Đặt giới hạn trục để chấm tròn không bị cắt
    ax.set_xlim(-0.5, NUM_CLIENTS - 0.5)
    ax.set_ylim(min(classes) - 0.5, max(classes) + 0.5)
    
    plt.tight_layout()
    
    save_path = os.path.join(core_dir, "data_split", "distribution_plot.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n→ Đã lưu biểu đồ 2D bubble chart tại: {os.path.basename(save_path)}")
    print(f"   • Trục X (ngang): Client ID")
    print(f"   • Trục Y (dọc): Class Label")
    print(f"   • Kích thước chấm tròn: Tỷ lệ với số lượng mẫu")
    print(f"   • Số trên chấm: Số lượng mẫu chính xác")
    plt.close()

if __name__ == "__main__":
    print("=" * 60)
    print("FL-IDS DATA PREPARATION PIPELINE")
    print("=" * 60)
    
    # 1. Load và clean dữ liệu
    X_raw, y = load_and_clean_data(FILE_PATH)
    
    if X_raw is None:
        print("Lỗi: Không thể load dữ liệu!")
        exit(1)
    
    # 2. Feature Engineering
    X_final, y_final, selected_features, scaler = feature_engineering(
        X_raw, y, threshold=FEATURE_SELECTION_THRESHOLD
    )
    
    # 3. Chia Train/Test TRƯỚC KHI chia cho clients
    X_train, X_test, y_train, y_test = split_train_test(
        X_final, y_final, train_ratio=TRAIN_RATIO
    )
    
    # 4. Lưu Global Test Set
    save_global_test_set(X_test, y_test)
    
    # 5. Chia Train set cho các Clients (Non-IID)
    stats = partition_data_non_iid(X_train, y_train, num_clients=NUM_CLIENTS, alpha=ALPHA)
    
    # 6. Vẽ biểu đồ phân phối
    visualize_distribution(stats)
    
    print("\n" + "=" * 60)
    print("✓ HOÀN TẤT! Dữ liệu đã được chuẩn bị.")
    print("=" * 60)
    print(f"→ Số features cuối cùng: {X_final.shape[1]} (từ {X_raw.shape[1]} features ban đầu)")
    print(f"→ Train set: {len(X_train)} samples (chia cho {NUM_CLIENTS} clients)")
    print(f"→ Test set: {len(X_test)} samples (dùng cho server evaluation)")
    print("=" * 60)
