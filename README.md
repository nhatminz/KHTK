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

Chạy riêng TabPFN v2 trên CPU, một estimator, dùng **toàn bộ 2.016 dòng train** làm ngữ cảnh; tải checkpoint ở lần đầu. Validation/test được giữ riêng. Nếu bổ sung dữ liệu thật vào train, TabPFN cũng dùng toàn bộ những dòng đó:

```powershell
.\.venv\Scripts\python.exe huan_luyen_yaris_ml.py --models TabPFN --tabpfn-version v2 --tabpfn-rows 0 --skip-ablations --skip-profile-experiment --skip-permutation-importance --output-dir outputs/tabpfn_full
```

Muốn so sánh trong cùng một lần chọn model:

```powershell
.\.venv\Scripts\python.exe huan_luyen_yaris_ml.py --models LogisticRegression RandomForest HistGradientBoosting MLP TabPFN --search-iterations 2 --skip-ablations --skip-profile-experiment --skip-permutation-importance --output-dir outputs/comparison
```

Trên macOS/Linux, thay `.\.venv\Scripts\python.exe` trong các lệnh bằng `python`. TabPFN không tự chạy trong lệnh mặc định; chọn qua `--models TabPFN`. `--tabpfn-rows 0` là mặc định mới, nghĩa là không lấy mẫu giảm số dòng; có thể bỏ tùy chọn này. Code cho phép chạy toàn bộ train vượt giới hạn CPU mặc định của thư viện. Bản v2 mặc định không dùng luồng đăng nhập checkpoint mới. Bản `--tabpfn-version v3.5-fast` có thể cần quyền truy cập/token PriorLabs; code không tự mở trình duyệt đăng nhập. [Hướng dẫn TabPFN](https://github.com/PriorLabs/TabPFN).

Hai mẫu đã chuẩn bị, chưa chứa quan sát thật:

- `data/real/measurements_template.csv`: ghi các số đo tại xưởng.
- `data/real/training_template.csv`: nhập đầu vào và hai nhãn chuyên môn độc lập; xem tên cột/phạm vi/cách ghi bằng chứng trong `data/real/entry_schema.json`.

Sao chép mẫu để nhập dữ liệu. Giữ cùng `vehicle_id` cho mọi phiên của cùng xe; dùng ID xe thật khác ID giả lập, ghi `data_origin=REAL_MEASURED`. Không tự điền giá trị chưa đo hoặc nhãn chưa biết.

```powershell
Copy-Item data/real/training_template.csv data/real/training_measured.csv
# Điền dữ liệu vào training_measured.csv trước khi chạy hai lệnh dưới.
.\.venv\Scripts\python.exe prepare_real_data.py --validate data/real/training_measured.csv
.\.venv\Scripts\python.exe huan_luyen_yaris_ml.py --additional-train-data data/real/training_measured.csv --models MLP --search-iterations 2 --skip-ablations --skip-profile-experiment --skip-permutation-importance --output-dir outputs/nn_with_real
```

Dữ liệu thật bổ sung chỉ được đưa vào train sau khi chia dữ liệu gốc theo xe. Validation/test gốc giữ nguyên và vẫn là giả lập. File mẫu trống chưa làm tăng số dòng train. Muốn tạo bộ mẫu khác mà không ghi đè mẫu đã nhập:

```powershell
.\.venv\Scripts\python.exe prepare_real_data.py --output-dir data/real/new_batch
```

Dự đoán bằng pipeline đã lưu, chọn đúng thư mục model:

```powershell
.\.venv\Scripts\python.exe predict.py --input outputs/nn/reports/example_input.csv --models-dir outputs/nn/models --output outputs/nn/predictions.csv
.\.venv\Scripts\python.exe predict.py --input outputs/tabpfn_full/reports/example_input.csv --models-dir outputs/tabpfn_full/models --output outputs/tabpfn_full/predictions.csv
```

Tra cứu các tùy chọn:

```powershell
.\.venv\Scripts\python.exe huan_luyen_yaris_ml.py --help
.\.venv\Scripts\python.exe predict.py --help
.\.venv\Scripts\python.exe prepare_real_data.py --help
```

# Kết quả

Dữ liệu gốc: `data/raw/du_lieu_phan_loai_gia_lap.csv`, 2.880 phiên giả lập, 360 xe; train/validation/test = 2.016/432/432 dòng. Chưa thu thập thêm quan sát xe thật; đã tạo hai mẫu nhập trống. Ba file raw giữ nguyên nội dung.

Kết quả đã có trước lần bổ sung MLP/TabPFN, model chọn bằng validation:

| Nhãn | Model | Test AP | Test F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Y1 | HistGradientBoosting + engineered | 0,8911 | 0,7801 | 0,8785 | 0,7015 |
| Y2 | LogisticRegression raw | 0,6357 | 0,5169 | 0,4107 | 0,6970 |

Kết quả lần chạy trước trên cùng split gốc, seed 2026. MLP chọn từ bản raw/engineered bằng validation; các dòng TabPFN trong bảng dưới thuộc **cấu hình cũ 512 dòng train**, chưa phải kết quả toàn bộ train:

| Phương pháp | Nhãn | Biến thể | Threshold | Test Accuracy | Test AP | Test F1 | Precision | Recall | FPR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | Y1 | engineered | 0,9267 | 0,8773 | 0,9088 | 0,7922 | 0,8347 | 0,7537 | 0,0671 |
| MLP | Y2 | raw | 0,2289 | 0,9259 | 0,5339 | 0,5556 | 0,5128 | 0,6061 | 0,0476 |
| TabPFN v2 | Y1 | raw | 0,4665 | 0,7222 | 0,7199 | 0,1892 | 1,0000 | 0,1045 | 0,0000 |
| TabPFN v2 | Y2 | raw | 0,1275 | 0,9259 | 0,7315 | 0,5676 | 0,5122 | 0,6364 | 0,0501 |

MLP hoàn tất trong 58,54 giây; TabPFN trong 47,94 giây trên máy hiện tại, không gồm thời gian cài thư viện. MLP có cảnh báo một số lượt fit chạm giới hạn 200 vòng và chưa hội tụ; giữ giới hạn để chạy nhẹ. TabPFN có cảnh báo CPU trên 200 mẫu; mỗi nhãn dùng 512 dòng, một estimator. TabPFN Y1 có Recall test thấp dù AP validation 0,8916; ngưỡng giữ nguyên từ validation, không chỉnh lại theo test. MLP Y2 cũng giảm AP từ 0,9580 trên validation xuống 0,5339 trên test.

Đã cài `tabpfn==9.1.0` và `torch==2.14.1` trong `.venv`, chạy hai phương pháp và lưu/nạp lại pipeline thành công. Bộ tests không chạy lại theo yêu cầu trước đó; tests mới cho nhập dữ liệu thật đã được viết nhưng chưa chạy. Chưa có kết quả train với dữ liệu thật vì bạn chưa nhập số đo/nhãn vào các mẫu.

Mỗi thư mục output có `reports/model_comparison.csv` (mọi candidate), `reports/evaluation.json` (model đã chọn), `reports/training_config.json` (cấu hình/phiên bản), `models/` (pipeline/ngưỡng), `plots/` (confusion matrix/PR/ROC). Lần chạy cũ ở `outputs/tabpfn/` dùng 512 dòng; lệnh mới ghi vào `outputs/tabpfn_full/` và dùng toàn bộ train. Config ghi `actual_train_context_rows` để kiểm tra số dòng thực tế. Khi nạp lại, wrapper khôi phục model từ ngữ cảnh đã lưu và checkpoint cache.

Các điểm số trên dữ liệu giả lập không xác nhận khả năng chẩn đoán xe thật hoặc nhu cầu nâng cấp máy phát. Y2 baseline có AP validation 0,9348 nhưng AP test 0,6357; lựa chọn/ngưỡng giữ theo validation, không đổi theo test.

Lần chạy mới **TabPFN dùng toàn bộ 2.016 dòng train** đã hoàn tất trong 126,05 giây. Đã nạp hai pipeline đã lưu và xác nhận mỗi nhãn giữ đúng 2.016 dòng ngữ cảnh, không lấy mẫu. Kết quả tại `outputs/tabpfn_full/`:

| Nhãn | Dòng train | Threshold | Test Accuracy | Test AP | Test F1 | Precision | Recall | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Y1 | 2.016 | 0,3744 | 0,7338 | 0,7099 | 0,2857 | 0,8519 | 0,1716 | 0,0134 |
| Y2 | 2.016 | 0,1998 | 0,9514 | 0,8139 | 0,7123 | 0,6500 | 0,7879 | 0,0351 |

So với cấu hình 512 dòng, F1 test tăng ở cả hai nhãn; AP test Y2 tăng, AP test Y1 giảm nhẹ. Toàn bộ train không bảo đảm mọi metric đều tăng. Validation/test vẫn giữ riêng; không đưa toàn bộ 2.880 dòng vào ngữ cảnh trong lượt đánh giá này. `reports/context_rows_verified.json` ghi số dòng đã xác nhận trong hai pipeline lưu trên đĩa.
