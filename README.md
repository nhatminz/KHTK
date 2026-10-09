# KHTK: uoc duoc thay dep trai cho A+ <3
# Hướng dẫn chạy

Windows PowerShell, tại thư mục dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tabpfn.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tabpfn.txt
```

Train các model CPU, gồm neural network MLP nhẹ (32–16 nút, tối đa 200 vòng):

```powershell
.\.venv\Scripts\python.exe huan_luyen_yaris_ml.py
```

Chạy riêng MLP, giữ kết quả ở `outputs/nn/`:

```powershell
.\.venv\Scripts\python.exe huan_luyen_yaris_ml.py --models MLP --search-iterations 2 --skip-ablations --skip-profile-experiment --skip-permutation-importance --output-dir outputs/nn
```

Chạy riêng TabPFN v2 trên CPU, một estimator, dùng **toàn bộ tập train** làm ngữ cảnh; tải checkpoint ở lần đầu. Với dataset mở rộng 8.640 dòng, split mặc định dùng 6.048 dòng train và 1.296 dòng cho mỗi tập validation/test. Nếu bổ sung dữ liệu thật vào train, TabPFN cũng dùng toàn bộ những dòng đó:

```powershell
.\.venv\Scripts\python.exe huan_luyen_yaris_ml.py --models TabPFN --tabpfn-version v2 --tabpfn-rows 0 --skip-ablations --skip-profile-experiment --skip-permutation-importance --output-dir outputs/tabpfn_expanded
```

Muốn so sánh trong cùng một lần chọn model:

```powershell
.\.venv\Scripts\python.exe huan_luyen_yaris_ml.py --models LogisticRegression RandomForest HistGradientBoosting MLP TabPFN --search-iterations 2 --skip-ablations --skip-profile-experiment --skip-permutation-importance --output-dir outputs/comparison
```

Trên macOS/Linux, thay `.\.venv\Scripts\python.exe` trong các lệnh bằng `python`. TabPFN không tự chạy trong lệnh mặc định; chọn qua `--models TabPFN`. `--tabpfn-rows 0` là mặc định mới, nghĩa là không lấy mẫu giảm số dòng; có thể bỏ tùy chọn này. Code cho phép chạy toàn bộ train vượt giới hạn CPU mặc định của thư viện. Bản v2 mặc định không dùng luồng đăng nhập checkpoint mới. Bản `--tabpfn-version v3.5-fast` có thể cần quyền truy cập/token PriorLabs; code không tự mở trình duyệt đăng nhập. [Hướng dẫn TabPFN](https://github.com/PriorLabs/TabPFN).


```powershell
Copy-Item data/real/training_template.csv data/real/training_measured.csv
# Điền dữ liệu vào training_measured.csv trước khi chạy hai lệnh dưới.
.\.venv\Scripts\python.exe prepare_real_data.py --validate data/real/training_measured.csv
.\.venv\Scripts\python.exe huan_luyen_yaris_ml.py --additional-train-data data/real/training_measured.csv --models MLP --search-iterations 2 --skip-ablations --skip-profile-experiment --skip-permutation-importance --output-dir outputs/nn_with_real
```

Dữ liệu thật bổ sung chỉ được đưa vào train sau khi chia dữ liệu gốc theo xe. Validation/test gốc giữ nguyên và vẫn là giả lập. Muốn tạo bộ mẫu khác mà không ghi đè mẫu đã nhập:

```powershell
.\.venv\Scripts\python.exe prepare_real_data.py --output-dir data/real/new_batch
```

Dự đoán bằng pipeline đã lưu, chọn đúng thư mục model:

```powershell
.\.venv\Scripts\python.exe predict.py --input outputs/nn/reports/example_input.csv --models-dir outputs/nn/models --output outputs/nn/predictions.csv
.\.venv\Scripts\python.exe predict.py --input outputs/tabpfn_expanded/reports/example_input.csv --models-dir outputs/tabpfn_expanded/models --output outputs/tabpfn_expanded/predictions.csv
```

Tra cứu các tùy chọn:

```powershell
.\.venv\Scripts\python.exe huan_luyen_yaris_ml.py --help
.\.venv\Scripts\python.exe predict.py --help
.\.venv\Scripts\python.exe prepare_real_data.py --help
```

```powershell
.\.venv\Scripts\python.exe huan_luyen_yaris_ml.py --models MLP --search-iterations 2 --skip-ablations --skip-profile-experiment --skip-permutation-importance --output-dir outputs/nn_expanded
```

Kết quả đã có trước lần bổ sung MLP/TabPFN, model chọn bằng validation:

| Nhãn | Model | Test AP | Test F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Y1 | HistGradientBoosting + engineered | 0,8911 | 0,7801 | 0,8785 | 0,7015 |
| Y2 | LogisticRegression raw | 0,6357 | 0,5169 | 0,4107 | 0,6970 |

Kết quả lần chạy trên cùng split gốc, seed 2026. MLP chọn từ bản raw/engineered bằng validation; các dòng TabPFN trong bảng dưới thuộc **cấu hình 512 dòng train**:

| Phương pháp | Nhãn | Biến thể | Threshold | Test Accuracy | Test AP | Test F1 | Precision | Recall | FPR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | Y1 | engineered | 0,9267 | 0,8773 | 0,9088 | 0,7922 | 0,8347 | 0,7537 | 0,0671 |
| MLP | Y2 | raw | 0,2289 | 0,9259 | 0,5339 | 0,5556 | 0,5128 | 0,6061 | 0,0476 |
| TabPFN v2 | Y1 | raw | 0,4665 | 0,7222 | 0,7199 | 0,1892 | 1,0000 | 0,1045 | 0,0000 |
| TabPFN v2 | Y2 | raw | 0,1275 | 0,9259 | 0,7315 | 0,5676 | 0,5122 | 0,6364 | 0,0501 |


**TabPFN dùng toàn bộ 2.016 dòng train**:

| Nhãn | Dòng train | Threshold | Test Accuracy | Test AP | Test F1 | Precision | Recall | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Y1 | 2.016 | 0,3744 | 0,7338 | 0,7099 | 0,2857 | 0,8519 | 0,1716 | 0,0134 |
| Y2 | 2.016 | 0,1998 | 0,9514 | 0,8139 | 0,7123 | 0,6500 | 0,7879 | 0,0351 |

# Kỹ thuật xử lý dữ liệu

- Kiểm tra schema, ID trùng/thiếu, nhãn 0/1 và phạm vi số; giữ dòng thiếu nhiên liệu và outlier hợp lệ, không tự xóa theo IQR.
- Numeric: median imputation và missing indicators; categorical: điền `<MISSING>` và OneHotEncoder xử lý category mới. StandardScaler dùng cho LogisticRegression/MLP. TabPFN dùng đầu vào native, không OHE/scaling bên ngoài.
- Chỉ fit preprocessing trên train/training fold. Chia train/validation/test theo `vehicle_id` gần 70/15/15, tuning bằng GroupKFold; chọn model/ngưỡng trên validation.
- Loại ID, hai target và biến hậu nghiệm gây leakage khỏi input: dư địa điện, tổng tải mới, dòng máy phát khả dụng, điện áp mô phỏng và chết máy. Ca sĩ/thể loại chỉ ở ablation riêng.
- Feature engineering: tổng tải nền/đèn/quạt/sấy kính, volume × RMS, SOH × tổng tải, RPM × tỷ lệ không tải; không tái tạo nhãn từ biến hậu nghiệm.
- Dữ liệu thêm có ID xe/scenario mới, tính lại dòng điện và nhãn theo quy tắc v2; giữ schema 43 cột và lưu nguồn batch ngoài CSV. Không sao chép nhãn cũ hoặc nhân bản các dòng cũ thành xe mới.
