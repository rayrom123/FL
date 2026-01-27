import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#model de xu ly, luu lai dung ban dang fit voi data nay
def load_and_preprocess_ciciot(file_path):
    print(f"--- Đang xử lý file: {file_path} ---")
    
    # 1. Load dữ liệu
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return None, None

    # 2. XÓA CỘT RÁC
    if 'Number' in df.columns:
        print("-> Đã xóa cột 'Number'.")
        df.drop(columns=['Number'], inplace=True)
    
    # 3. Xử lý Inf/NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"-> Phát hiện {missing_count} giá trị lỗi (NaN/Inf). Đang xóa dòng...")
        df.dropna(inplace=True)

    # 4. MAPPING NHÃN
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
    
    # 5. Scale Features
    X_raw = df.drop(columns=['label', 'label_code'])
    print(f"-> Số lượng Features giữ lại: {X_raw.shape[1]}")
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    print(f"-> Hoàn tất chuẩn hóa. Output Shape: {X_scaled.shape}")
    
    return X_scaled, y

if __name__ == "__main__":
    pass
