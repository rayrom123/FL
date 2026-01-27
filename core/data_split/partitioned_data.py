import os
import sys

# Thêm path để import dataset_preprocessor
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.dirname(current_dir)
if core_dir not in sys.path:
    sys.path.append(core_dir)

from dataset_preprocessor import load_and_preprocess_ciciot

# CẤU HÌNH
FILE_PATH = "d:/FL/core/dataset/cic-iot23.csv"
NUM_CLIENTS = 5
ALPHA = 0.5  # Độ lệch Non-IID. 
             # Alpha càng nhỏ (0.1) -> Càng Non-IID (mỗi client chỉ giữ 1 vài loại nhãn).
             # Alpha càng lớn (100) -> Càng giống IID (chia đều).

def partition_data_non_iid():
    # 1. Load dữ liệu đã sạch
    # X, y là numpy array
    X, y = load_and_preprocess_ciciot(FILE_PATH)
    
    n_samples = y.shape[0]
    n_classes = len(np.unique(y))
    
    # 2. Tạo Dictionary chứa index của từng class
    # class_indices[0] = [index của các dòng Benign]
    # class_indices[1] = [index của các dòng DDoS...]
    class_indices = [np.argwhere(y == k).flatten() for k in range(n_classes)]
    
    # Khởi tạo list chứa index cho từng client
    client_indices = [[] for _ in range(NUM_CLIENTS)]
    
    print(f"\n--- Đang chia Non-IID (Alpha={ALPHA}) ---")
    
    # 3. Thuật toán Dirichlet Partitioning
    for k in range(n_classes):
        idx_k = class_indices[k]
        np.random.shuffle(idx_k)
        
        # Tạo phân phối xác suất cho class k trên 5 clients
        # Ví dụ: Class DDoS -> [0.8, 0.1, 0.05, 0.05, 0.0] (Client 0 giữ 80% DDoS)
        proportions = np.random.dirichlet(np.repeat(ALPHA, NUM_CLIENTS))
        
        # Cân đối lại proportions để tránh trường hợp một client nhận 0 mẫu do làm tròn
        # Dùng np.split để chia mảng index dựa trên tỷ lệ
        proportions = np.array([p * (len(idx_k) < n_samples / NUM_CLIENTS) for p in proportions])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        idx_split = np.split(idx_k, proportions)
        
        # Gán index vào từng client
        for i in range(NUM_CLIENTS):
            client_indices[i] += idx_split[i].tolist()

    # 4. Lưu dữ liệu ra file .pt (Torch Tensor)
    # Tách 20% làm Global Test Set (Server giữ)
    # 80% còn lại chia cho Clients
    
    # Để đơn giản, ta sẽ lấy random 20% từ X tổng để làm Test set trước
    # Tuy nhiên, để đúng bài toán FL Non-IID, Test set nên bao gồm đủ các loại tấn công
    # Cách tốt nhất: Data từng Client giữ nguyên, Server lấy 1 phần nhỏ đại diện hoặc dùng tập Test riêng.
    
    # Ở đây tôi làm cách: Save toàn bộ data chia được vào file client.
    # Khi training, mỗi client tự cắt 20% của mình làm val_set, còn server sẽ test trên tập hợp dữ liệu test riêng.
    
    stats = []
    
    for i in range(NUM_CLIENTS):
        indices = client_indices[i]
        np.random.shuffle(indices) # Trộn lại lần cuối để lúc train không bị theo thứ tự class
        
        X_client = torch.tensor(X[indices], dtype=torch.float32)
        y_client = torch.tensor(y[indices], dtype=torch.long)
        
        save_path = f"client_{i}_data.pt"
        torch.save((X_client, y_client), save_path)
        
        # Thống kê label để vẽ biểu đồ
        unique, counts = np.unique(y[indices], return_counts=True)
        stats.append(dict(zip(unique, counts)))
        
        print(f"-> Client {i}: {len(indices)} mẫu. Đã lưu '{save_path}'")

    return stats
#file dataset chia 7/3 -> phan 7 chia cilent
def visualize_distribution(stats):
    # Vẽ biểu đồ phân phối để bạn thấy độ Non-IID
    classes = set()
    for s in stats:
        classes.update(s.keys())
    classes = sorted(list(classes))
    
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(NUM_CLIENTS)
    
    for cls in classes:
        counts = [s.get(cls, 0) for s in stats]
        plt.bar(range(NUM_CLIENTS), counts, bottom=bottom, label=f'Class {cls}')
        bottom += np.array(counts)
        
    plt.xlabel('Client ID')
    plt.ylabel('Number of Samples')
    plt.title(f'Non-IID Data Distribution (Alpha={ALPHA})')
    # plt.legend() # Legend sẽ rất dài vì 34 class, nên comment lại hoặc để outside
    plt.show()

if __name__ == "__main__":
    stats = partition_data_non_iid()
    visualize_distribution(stats)