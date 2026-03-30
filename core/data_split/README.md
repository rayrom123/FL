# Hướng dẫn Chuẩn bị và Phân chia Dữ liệu (Data Split)

Thư mục này chứa các script để xử lý, làm sạch và phân chia bộ dữ liệu **CIC-IoT23** cho các bài toán huấn luyện khác nhau.

## 1. Yêu cầu Tiền đề

- File dữ liệu gốc `cic-iot23.csv` phải được đặt tại: `d:\FL\core\dataset\cic-iot23.csv`.
- Môi trường Python cần cài đặt các thư viện: `pandas`, `numpy`, `torch`, `scikit-learn`, `matplotlib`.

---

## 2. Quy trình Thực hiện

### Bước 1: Chạy Script Phân chia Dữ liệu

Chạy file này để thực hiện: Làm sạch dữ liệu -> Feature Engineering -> Chia Task (CL) -> Chia Client (FL).

```bash
python partitioned_data.py
```

**Kết quả sau khi chạy:**

- Tạo thư mục `centralized_data/`: Chứa dữ liệu cho huấn luyện tập trung và Continual Learning.
- Tạo thư mục `federated_data/`: Chứa 30 file dữ liệu Non-IID cho 5 Clients tham gia Federated Continual Learning.
- File `30_test_data.pt`: Bộ dữ liệu kiểm tra (30% tổng số mẫu) dùng chung cho tất cả các mô hình.

### Bước 2: Tạo Biểu đồ Phân phối

Sau khi đã có dữ liệu ở Bước 1, chạy script này để quan sát trực quan cách dữ liệu được phân chia:

```bash
python visualize_data_split.py
```

**Kết quả:** Các biểu đồ PNG sẽ được lưu tại thư mục gốc của `data_split/bieu_do`.

---

## 3. Cấu trúc Thư mục Dữ liệu đầu ra

### 📁 `centralized_data/` (Dành cho Centralized & CL)

- `centralized_train_data.pt`: Toàn bộ 70% dữ liệu dùng để huấn luyện mô hình Server.
- `task_X_train.pt`: Dữ liệu huấn luyện riêng cho Task X (X từ 1 đến 6).
- `task_X_test.pt`: Dữ liệu kiểm tra riêng cho Task X.

### 📁 `federated_data/` (Dành cho Federated CL)

- Gồm 30 file có định dạng: `client{i}_task{T}.pt`.
- Ví dụ: `client0_task1.pt` là dữ liệu của Client 0 cho Task 1.

### 📄 File Test chung

- `30_test_data.pt`: Đây là bộ dữ liệu "vàng" dùng để đánh giá cuối cùng cho mọi kịch bản thí nghiệm.

---

## 4. Các kịch bản sử dụng

1. **Huấn luyện mô hình tập trung (Centralized):** Sử dụng `centralized_data/centralized_train_data.pt`.
2. **Huấn luyện Incremental/Continual:** Sử dụng các file `task_X_train.pt` theo thứ tự từ 1 đến 6.
3. **Huấn luyện Federated Continual Learning:** Sử dụng các file `client{i}_task{T}.pt` cho từng client tại mỗi vòng học task tương ứng.
