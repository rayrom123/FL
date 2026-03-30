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
FILE_PATH = os.path.join(core_dir, "dataset", "cic-iot23.csv")
NUM_CLIENTS = 5
ALPHA = 0.5  # Độ lệch Non-IID
TRAIN_RATIO = 0.7  # 70% train, 30% test
FEATURE_SELECTION_THRESHOLD = 0.001  # Ngưỡng để loại bỏ features không quan trọng
RANDOM_SEED = 42  # Fix seed để kết quả reproducible

# FIX RANDOM SEED cho tất cả thư viện
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# CẤU HÌNH TASK (Chia 34 nhãn thành 6 task: 6, 6, 6, 6, 5, 5)
TASK_CONFIG = {
    1: [0, 1, 24, 25, 26, 27],
    2: [2, 13, 11, 12, 29, 30],
    3: [3, 14, 16, 17, 18, 19],
    4: [4, 15, 20, 21, 22, 23],
    5: [5, 28, 8, 9, 10],
    6: [6, 7, 31, 32, 33]
}

# Thư mục lưu trữ dữ liệu đã xử lý
CENTRALIZED_DIR = os.path.join(core_dir, "data_split", "centralized_data")
FEDERATED_DIR = os.path.join(core_dir, "data_split", "federated_data")
os.makedirs(CENTRALIZED_DIR, exist_ok=True)
os.makedirs(FEDERATED_DIR, exist_ok=True)

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

def split_by_class_to_tasks(X, y, task_config, train_ratio=0.7):
    """
    Chia Train/Test cho từng class trước, rồi mới gộp lại thành từng task.
    Dữ liệu được lưu trong CENTRALIZED_DIR.
    """
    print(f"\n--- CHIA TRAIN/TEST PER-CLASS VÀ GỘP THÀNH {len(task_config)} TASKS ---")
    
    X_global_test_list = []
    y_global_test_list = []
    
    all_X_train_dict = {} # Lưu tensor theo task_id

    for task_id, labels in task_config.items():
        task_train_X, task_train_y = [], []
        task_test_X, task_test_y = [], []
        
        for label in labels:
            mask = (y == label)
            X_cls = X[mask]
            y_cls = y[mask]
            
            if len(y_cls) < 2:
                if len(y_cls) > 0:
                    task_train_X.append(X_cls)
                    task_train_y.append(y_cls)
                continue
                
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_cls, y_cls, train_size=train_ratio, random_state=42
            )
            
            task_train_X.append(X_tr)
            task_train_y.append(y_tr)
            task_test_X.append(X_te)
            task_test_y.append(y_te)
            
        if task_train_y:
            X_task_train = np.concatenate(task_train_X, axis=0)
            y_task_train = np.concatenate(task_train_y, axis=0)
            
            # Lưu Task Train vào CENTRALIZED_DIR
            train_path = os.path.join(CENTRALIZED_DIR, f"task_{task_id}_train.pt")
            torch.save((torch.tensor(X_task_train, dtype=torch.float32), 
                        torch.tensor(y_task_train, dtype=torch.long)), train_path)
            
            all_X_train_dict[task_id] = (X_task_train, y_task_train)
        
        if task_test_y:
            X_task_test = np.concatenate(task_test_X, axis=0)
            y_task_test = np.concatenate(task_test_y, axis=0)
            
            # Lưu Task Test vào CENTRALIZED_DIR
            test_path = os.path.join(CENTRALIZED_DIR, f"task_{task_id}_test.pt")
            torch.save((torch.tensor(X_task_test, dtype=torch.float32), 
                        torch.tensor(y_task_test, dtype=torch.long)), test_path)
            
            X_global_test_list.append(X_task_test)
            y_global_test_list.append(y_task_test)
            
        print(f"→ Task {task_id} (Centralized): Train={len(y_task_train) if task_train_y else 0}, Test={len(y_task_test) if task_test_y else 0}")

    # Gộp toàn bộ train data thành một file duy nhất cho Centralized training tổng quát
    all_X_train_full = np.concatenate([v[0] for v in all_X_train_dict.values()], axis=0)
    all_y_train_full = np.concatenate([v[1] for v in all_X_train_dict.values()], axis=0)
    centralized_train_path = os.path.join(CENTRALIZED_DIR, "centralized_train_data.pt")
    torch.save((torch.tensor(all_X_train_full, dtype=torch.float32), 
                torch.tensor(all_y_train_full, dtype=torch.long)), centralized_train_path)
    print(f"\n→ Đã lưu Unified Centralized Train Set: {len(all_y_train_full)} mẫu tại: {os.path.basename(centralized_train_path)}")

    # Gộp thành Global Test Set
    X_global_test = np.concatenate(X_global_test_list, axis=0)
    y_global_test = np.concatenate(y_global_test_list, axis=0)
    
    global_test_path = os.path.join(core_dir, "data_split", "30_test_data.pt")
    torch.save((torch.tensor(X_global_test, dtype=torch.float32), 
                torch.tensor(y_global_test, dtype=torch.long)), global_test_path)
    print(f"→ Đã lưu Global Test Set: {len(y_global_test)} mẫu tại: {os.path.basename(global_test_path)}")

    return all_X_train_dict, X_global_test, y_global_test

def partition_task_for_clients(X_task, y_task, task_id, num_clients=5, alpha=0.5):
    """
    Chia dữ liệu của MỘT TASK cho các clients (Non-IID).
    Lưu file theo định dạng client{i}_task{T}.pt.
    """
    n_samples = y_task.shape[0]
    unique_labels = np.unique(y_task)
    n_classes = len(unique_labels)
    
    # Dirichlet Partitioning cho Task này
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    class_indices = [np.argwhere(y_task == lbl).flatten() for lbl in unique_labels]
    client_indices = [[] for _ in range(num_clients)]
    
    for k in range(n_classes):
        idx_k = class_indices[k]
        if len(idx_k) == 0: continue
        np.random.shuffle(idx_k)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array([p * (len(idx_k) < n_samples / num_clients) for p in proportions])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        idx_split = np.split(idx_k, proportions)
        for i in range(num_clients):
            client_indices[i] += idx_split[i].tolist()

    for i in range(num_clients):
        indices = client_indices[i]
        if len(indices) == 0: continue
        np.random.shuffle(indices)
        
        X_client_task = torch.tensor(X_task[indices], dtype=torch.float32)
        y_client_task = torch.tensor(y_task[indices], dtype=torch.long)
        
        filename = f"client{i}_task{task_id}.pt"
        save_path = os.path.join(FEDERATED_DIR, filename)
        torch.save((X_client_task, y_client_task), save_path)
        
    print(f"   ✓ Đã chia Task {task_id} thành {num_clients} client files.")

if __name__ == "__main__":
    print("=" * 60)
    print("FL-IDS DATA PREPARATION PIPELINE (Federated Continual Learning)")
    print("=" * 60)
    
    # 1. Load và clean dữ liệu
    X_raw, y = load_and_clean_data(FILE_PATH)
    if X_raw is None: exit(1)
    
    # 2. Feature Engineering
    X_final, y_final, selected_features, scaler = feature_engineering(
        X_raw, y, threshold=FEATURE_SELECTION_THRESHOLD
    )
    
    # 3. Chia Per-Class và gộp thành Tasks
    task_train_dict, X_global_test, y_global_test = split_by_class_to_tasks(
        X_final, y_final, TASK_CONFIG, TRAIN_RATIO
    )
    
    # 4. Chia từng Task cho Clients (Federated Continual Learning)
    print(f"\n--- CHIA TỪNG TASK CHO {NUM_CLIENTS} CLIENTS (Non-IID, Alpha={ALPHA}) ---")
    for task_id, (X_t, y_t) in task_train_dict.items():
        partition_task_for_clients(X_t, y_t, task_id, NUM_CLIENTS, ALPHA)
    
    print("\n" + "=" * 60)
    print("✓ HOÀN TẤT! Dữ liệu đã được chuẩn bị.")
    print("=" * 60)
    print(f"→ Centralized Data: {CENTRALIZED_DIR}")
    print(f"→ Federated Data: {FEDERATED_DIR} (30 files: client0_task1.pt ... client4_task6.pt)")
    print(f"→ Test Set: 30_test_data.pt (dùng cho đánh giá CL và FL)")
    print("=" * 60)
