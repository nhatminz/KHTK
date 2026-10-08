# KHTK — Phân loại kịch bản điện ô tô giả lập

Dự án nghiên cứu hai bộ phân loại nhị phân độc lập về ảnh hưởng của hệ thống giải trí Android và ampli đến hệ thống điện, với ngữ cảnh Toyota Yaris 2008 và các cấu hình xe tham chiếu khác.

**Toàn bộ 2.880 quan sát huấn luyện là giả lập. Nhãn do quy tắc mô phỏng sinh ra; điểm số ML không phải độ chính xác chẩn đoán xe thật.**

- **Y1** (`y1_man_hinh_gop_phan_thieu_dien`): 1 nghĩa là màn hình Android và ampli đi kèm góp phần làm thiếu dòng so với hệ đầu CD cũ trong kịch bản mô phỏng; 0 nghĩa là không có tác động theo định nghĩa nhãn. Y1 không đồng nghĩa với chết máy.
- **Y2** (`y2_can_nang_cap_may_phat`): 1 nghĩa là quy tắc mô phỏng gợi ý cân nhắc tăng công suất sạc trong một số tình huống; 0 nghĩa là không cần theo quy tắc này. Y2 không phải quyết định nâng cấp đã được chứng minh trên xe thật.

## 1. Dữ liệu dùng để train

**Nguồn train duy nhất và mặc định:**

```text
data/raw/du_lieu_phan_loai_gia_lap.csv
```

CSV này có đủ hai nhãn: 2.880 dòng, 43 cột, 360 `vehicle_id`, 14 `profile_id`, mỗi xe có 8 phiên. Y1 dương ở 880 phiên (30,56%), Y2 dương ở 223 phiên (7,74%). Y2 mất cân bằng nên Accuracy riêng lẻ rất dễ gây hiểu nhầm.

`data/raw/thong_so_xe_nguon_thuc.csv` là bảng tham chiếu 14 cấu hình, chứa hãng/mẫu, năm, dung tích, nhiên liệu và nguồn thông số. Code chỉ kiểm tra tính nhất quán với CSV theo `profile_id`, không merge thêm biến và không train trên bảng thiếu nhãn này. Dòng máy phát mô phỏng có thể lệch mức tham chiếu; báo cáo ghi riêng các sai khác, không tự sửa thành giá trị catalog.

`data/raw/yaris_2008_dataset_classification.xlsx` là workbook tham khảo. Code đọc các sheet để kiểm tra schema, công thức và so sánh `Scenarios_SIM` với CSV, xuất `Data_Dictionary`. `Real_Data_Entry` là mẫu nhập quan sát thực nghiệm, chưa có dữ liệu xe thật. Không cộng các phiên Excel vào CSV và không nối ba file để train.

Ba file gốc đã chuyển vào `data/raw/` và được xác minh SHA-256 trước/sau. Bằng chứng tại `outputs/reports/raw_data_integrity.json`. Không có bản dataset gốc trùng lặp ở root. `data/processed/` để dành cho dữ liệu dẫn xuất sau này; pipeline hiện xử lý trong bộ nhớ.

Các cột mang hậu tố `_GIA_LAP` là giả định hoặc kết quả mô phỏng, không phải số đo đã xác minh của xe người dùng. Những cột không mang hậu tố đó cũng không biến 2.880 phiên thành dữ liệu thật. CSV được đọc bằng UTF-8 BOM (`utf-8-sig`). Audit kiểm tra ký tự thay thế Unicode `�` trong các trường chuỗi; dữ liệu hiện tại không có ký tự này. Không tự đoán hoặc sửa nội dung dữ liệu gốc.

### Data dictionary và đơn vị

Dictionary gốc được trích xuất sang `outputs/eda/data_dictionary.csv`. Phân tích toàn bộ 43 cột nằm ở `outputs/reports/selected_features.json`, với thời điểm có thể biết giá trị, nguồn gốc, quan hệ với quy tắc nhãn và lý do giữ/loại. Workbook không cung cấp mã nguồn đầy đủ của bộ sinh nhãn: những quan hệ chưa xác minh được được ghi là giả định, không tuyên bố đã biết công thức chính xác.

### Bộ SAFE_FEATURES mặc định

Baseline giả định đã biết thông số thiết bị, điều kiện vận hành và các phép đo/ước lượng độc lập **trước quyết định về nhãn**. Từ “safe” chỉ nói về điều kiện sử dụng biến, không bảo đảm ứng dụng chẩn đoán thực tế. Nếu không có các đầu vào này, không được lấy giá trị mô phỏng thay cho số đo của xe.

| Biến | Ý nghĩa / đơn vị |
| --- | --- |
| `may_phat_dinh_muc_A_GIA_LAP` | Dòng định mức giả định, A; cần đối chiếu đúng thiết bị/VIN khi có dữ liệu thật |
| `suc_khoe_may_phat_pct_GIA_LAP` | Khả năng máy phát còn đạt so với chuẩn, %; cần phép đo độc lập |
| `suc_khoe_acquy_pct_GIA_LAP` | SOH ắc quy, % |
| `muc_sac_acquy_SOC_pct_GIA_LAP` | SOC, %; khác SOH |
| `ty_le_dung_cho_no_may_do_thi` | Tỷ lệ không tải của hành trình dự kiến, 0–1 |
| `vong_tua_may_rpm` | Vòng tua tại kịch bản, rpm |
| `tai_dien_co_ban_A` | Tải nền ECU/đánh lửa/bơm…, A |
| `den_pha_A` | Dòng đèn đang sử dụng, A |
| `quat_gio_dieu_hoa_A` | Dòng quạt gió, không phải toàn bộ công suất lạnh, A |
| `suoi_kinh_A` | Dòng sấy kính, A |
| `dong_sac_acquy_A` | Dòng sạc bình được coi là phụ tải trong mô phỏng, A; cần ước lượng trước độc lập |
| `dau_CD_cu_A` | Dòng đầu CD ở điều kiện đối chứng, A |
| `man_hinh_android_ranh_A` | Dòng màn hình lúc rảnh, A |
| `am_luong_pct` | Volume tương đối, 0–100%; không tuyến tính với công suất |
| `do_nang_luong_am_thanh_RMS_0_1` | Proxy năng lượng tín hiệu, 0–1 |
| `amp_ngoai_co_lap` | Có ampli ngoài, 0/1; xử lý dạng categorical trong pipeline |
| `amp_ngoai_dinh_muc_dong_A_GIA_LAP` | Dòng định mức ampli, A; không phải watt đầu ra loa |

Không tính trực tiếp cân bằng cấp dòng, chênh lệch tổng tải với máy phát hay công thức sinh nhãn. Thành phần đầu vào vẫn có thể là biến thượng nguồn của quy tắc; model có thể học lại giả định mô phỏng. Đây là một giới hạn cần công khai.

### Các biến bị loại và thí nghiệm riêng

| Biến / nhóm | Chính sách và lý do |
| --- | --- |
| `scenario_id`, `vehicle_id`, `profile_id` | Chỉ định danh/audit; `vehicle_id` dùng chia nhóm, `profile_id` dùng kiểm tra cấu hình; không đưa vào model |
| `du_dia_dien_moi_A_GIA_LAP`, `du_dia_dien_dau_cu_A_GIA_LAP` | Cân bằng điện mới/đối chứng, A; mã hóa trực tiếp cơ chế nhãn; luôn loại |
| `tong_tai_dien_moi_A_GIA_LAP` | Tổng tải mới, A; tính toán hậu nghiệm của mô phỏng; luôn loại |
| `dong_may_phat_co_san_A_GIA_LAP` | Dòng khả dụng tính theo công thức giả lập, A; luôn loại |
| `dien_ap_khi_no_may_V_GIA_LAP` | Điện áp đáp ứng được dẫn xuất cơ học, V; luôn loại dù điện áp thật đo trước chẩn đoán có thể hợp lệ trong nghiên cứu khác |
| `chet_may_GIA_LAP` | Kết quả chết máy, 0/1, xảy ra sau diễn biến; luôn loại |
| `loi_bom_xang_GIA_LAP` | Lỗi bơm xăng, 0/1; cơ chế chết máy độc lập, gây nhiễu mục tiêu điện; luôn loại |
| `tai_man_hinh_luc_phat_A_GIA_LAP`, `amp_ngoai_dong_thuc_A_GIA_LAP` | Dòng vận hành, A; chỉ thử trong bộ **measured** nếu đo độc lập trước gán nhãn; nguy cơ shortcut do dòng mô phỏng sinh cùng quy tắc |
| `loi_lap_dat_dien_GIA_LAP` | Lỗi lắp đặt, 0/1; bộ measured, chỉ khi đã kiểm tra trước quyết định; không suy từ nhãn |
| `tai_roi_nguon_do_loi_A` | Tải bất thường do lỗi, A; bộ measured, chỉ khi đã đo/ước lượng độc lập |
| `sut_ap_mat_V_GIA_LAP` | Sụt áp đường mát, V; bộ measured, đo trước quyết định |
| `the_loai_nhac`, `ca_si_minh_hoa` | Bộ **music** để ablation; không gán quan hệ nhân quả với điện tiêu thụ |
| `hang_xe`, `mau_xe` | Bộ **context** để kiểm tra tương quan giả/ghi nhớ cấu hình |
| `doi_xe`, `dung_tich_dong_co_L` | Năm và dung tích L, bộ context; năm hiện hằng số 2008, dung tích không phải công suất máy phát |
| `xang_do_thi_L_100km`, `xang_hon_hop_L_100km` | L/100 km, bộ context; quan hệ yếu và có 208 missing mỗi cột |
| `toc_do_xe_kmh` | km/h, bộ context; có thể dư thừa với trạng thái vòng tua |

Các bộ music/context/measured là thí nghiệm validation riêng, không đủ điều kiện trở thành model chính và không được đánh giá trên test. Chúng giữ nguyên thuật toán và hyperparameter của baseline raw tương ứng để so sánh ảnh hưởng của bộ biến. Permutation importance của context/music cũng được lưu để xem nhiên liệu, tên xe, ca sĩ có mang thông tin giả không.

## 2. Cài đặt

Khuyến nghị Python 3.11–3.13; phiên kiểm tra thực tế dùng Python 3.13.9. Dùng môi trường ảo riêng, không cần GPU.

Windows PowerShell, chạy từ thư mục dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python huan_luyen_yaris_ml.py
python predict.py --input outputs/reports/example_input.csv --output outputs/predictions.csv
```

Nếu chính sách PowerShell không cho kích hoạt, có thể chạy trực tiếp interpreter, không cần đổi chính sách hệ thống:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe huan_luyen_yaris_ml.py
.\.venv\Scripts\python.exe predict.py --input outputs/reports/example_input.csv --output outputs/predictions.csv
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python huan_luyen_yaris_ml.py
python predict.py --input outputs/reports/example_input.csv --output outputs/predictions.csv
```

XGBoost tùy chọn, chỉ được đưa vào danh sách so sánh khi import được:

```bash
python -m pip install xgboost==3.0.5
python huan_luyen_yaris_ml.py
```

Nếu không có XGBoost, chương trình ghi rõ việc bỏ qua trong `training_config.json`, vẫn chạy bốn thuật toán bắt buộc của scikit-learn. Lệnh `--models XGBoost` khi thư viện không có sẽ báo lỗi, không âm thầm đổi thuật toán. Chỉ load artifact joblib từ nguồn tin cậy và dùng cùng các phiên bản thư viện đã train; phiên bản thực tế được lưu trong config.

## 3. Chạy EDA, train và cấu hình

Lệnh tối giản giữ tương thích:

```bash
python huan_luyen_yaris_ml.py
python huan_luyen_yaris_ml.py --help
python predict.py --help
```

Ví dụ tùy chỉnh:

```bash
python huan_luyen_yaris_ml.py --eda-only
python huan_luyen_yaris_ml.py --data data/raw/du_lieu_phan_loai_gia_lap.csv --output-dir outputs --seed 2026 --cv-folds 3 --search-iterations 4 --jobs 2 --max-fpr 0.10
python huan_luyen_yaris_ml.py --models LogisticRegression RandomForest HistGradientBoosting --skip-ablations --skip-profile-experiment
```

`--data` nhận CSV theo đủ schema gốc; tham chiếu CSV/Excel được tìm trong cùng thư mục để audit nếu có. `--output-dir` là nơi lưu kết quả. `--seed` mặc định 2026. `--search-iterations` giới hạn số cấu hình của RandomizedSearchCV mỗi thuật toán. `--cv-folds` là số fold GroupKFold trên train. `--jobs` giới hạn số luồng CPU, search chạy tuần tự. Dummy luôn được giữ khi chọn danh sách `--models`. `--max-fpr` là giới hạn FPR trên validation cho việc chọn threshold, phải xác định trước khi xem test.

Mỗi lần train có thể ghi đè artifacts trong thư mục output đã chọn. Để lưu nhiều thí nghiệm, đặt `--output-dir outputs/experiment_02` và truyền đúng `--models-dir` khi predict. Không đặt output trong `data/raw/`. Việc đổi seed hoặc cấu hình sau khi xem test có thể làm test trở thành dữ liệu lựa chọn; không dùng lặp thử test để tối ưu kết quả.

### Data quality và cleaning

- Audit toàn bộ schema/dtype, missing số lượng và %, duplicate hàng/`scenario_id`, ID thiếu và ID lặp. Xe và profile lặp là cấu trúc nhóm hợp lệ, khác với scenario trùng.
- Kiểm tra ánh xạ một xe–một profile, metadata so với reference, tỷ lệ nhãn và tổ hợp nhãn, nhãn theo vehicle/profile, phân bố numeric/category và correlation.
- Kiểm tra % trong [0,100], tỷ lệ/RMS trong [0,1], các cờ đúng 0/1, RPM dương, dòng/tốc độ không âm, NaN/Inf và chuỗi không phải số. Dư địa điện âm hợp lệ nên không bị coi là lỗi.
- Chỉ loại whitespace bao quanh và coi ô trống là missing trong bộ nhớ. Dữ liệu gốc không bị sửa. Dừng train nếu ID/nhãn/range sai hoặc scenario trùng để tránh sửa âm thầm.
- IQR chỉ gắn cờ outlier, không xóa hoặc clip: tải lớn và sự kiện hiếm có thể là kịch bản có chủ đích. IQR bằng 0 cũng có thể gắn cờ cờ nhị phân hợp lệ.
- Không xóa 208 dòng thiếu nhiên liệu. Baseline không dùng nhiên liệu; ablation context dùng imputer.

### Encoding và missing values

Mỗi model có Pipeline riêng: `FeatureBuilder → ColumnTransformer → classifier`. Numeric dùng median `SimpleImputer`, thêm missing indicator và giữ cột hoàn toàn trống trong train fold. LogisticRegression thêm StandardScaler; các cây không scale. Categorical dùng constant `<MISSING>` và `OneHotEncoder(handle_unknown="ignore")`, không LabelEncoder và không học thứ tự giả. Baseline encode cờ có ampli; các ablation hỗ trợ hãng, mẫu, thể loại, ca sĩ và cờ lỗi. Cardinality hiện thấp nên không cần target encoding.

Imputer, scaler và encoder đều nằm trong Pipeline, chỉ fit trên train của từng CV fold hoặc train chính. Validation/test chỉ transform. Không oversampling hoặc SMOTE. Model cuối giữ đúng fit trên train đã đánh giá validation; không refit với validation sau khi chọn ngưỡng, tránh thay đổi phân bố xác suất của model đã chọn.

### Feature engineering

Mỗi thuật toán thật được so sánh hai bản raw/engineered trên cùng train/validation, với cùng hyperparameter đã chọn từ CV raw. Không tìm thêm tham số cho bản engineered; cột CV-AP để trống ở bản này để không nhầm điểm raw là điểm CV của engineered.

| Feature | Công thức | Đơn vị / giả định |
| --- | --- | --- |
| `phu_tai_co_ban_tong_A` | `tai_dien_co_ban_A + den_pha_A + quat_gio_dieu_hoa_A + suoi_kinh_A` | A; tải thành phần biết trước; không cộng tải audio mới hay trừ dòng cấp khả dụng |
| `volume_RMS_proxy` | `(am_luong_pct / 100) * do_nang_luong_am_thanh_RMS_0_1` | Không thứ nguyên; proxy tương tác, không phải luật công suất tuyến tính |
| `SOH_phu_tai_A` | `(suc_khoe_acquy_pct_GIA_LAP / 100) * phu_tai_co_ban_tong_A` | A theo tương tác, không phải dòng đo; SOH cần biết độc lập |
| `rpm_do_thi_proxy` | `vong_tua_may_rpm * ty_le_dung_cho_no_may_do_thi` | rpm theo tương tác; điều kiện kịch bản biết trước |

Nếu một thành phần thiếu, feature dẫn xuất cũng thiếu, rồi được impute trong train fold. Không tự coi thành phần thiếu bằng 0. Không tạo chênh lệch dòng đầu Android đang phát với đầu cũ vì dòng đang phát không có trong bộ safe.

### Split, tuning và lựa chọn

1. GroupShuffleSplit hai bước theo `vehicle_id`, xác định số nhóm holdout bằng `round(0.15 * n_vehicles)`. Dữ liệu hiện có tách đúng train 252 xe/2.016 dòng, validation 54 xe/432 dòng, test 54 xe/432 dòng. Không có xe chung giữa ba tập.
2. Xác minh cả hai lớp của hai nhãn trong mỗi tập và CV fold. Nếu thiếu lớp, dừng với lỗi để thiết kế lại phân chia nhóm có giải thích; không tự dò seed để đạt test đẹp hơn. `split_summary.json` ghi tỷ lệ nhãn thực tế sau chia.
3. DummyClassifier (`prior`), LogisticRegression, RandomForest, HistGradientBoosting và XGBoost nếu có. Mỗi Y có classifier riêng.
4. RandomizedSearchCV trên train với GroupKFold theo vehicle, scoring Average Precision. RF tune số cây/độ sâu/min leaf/class weight; HGB tune số vòng, learning rate, số lá, L2, class weight; logistic tune C/class weight. HGB tắt internal early stopping để tránh tạo split theo hàng ẩn.
5. XGBoost dùng `binary:logistic`, `tree_method=hist`, tune số cây/độ sâu/learning rate/subsample/colsample. Wrapper cân bằng tính `scale_pos_weight = n_negative / n_positive` ngay trong `fit` của fold khi cấu hình yêu cầu, không lấy tỷ lệ từ validation/test.
6. Với mỗi candidate, threshold tối ưu trên validation: tối đa F1 dưới giới hạn FPR, tie-break bằng Recall, FPR thấp và gần 0.5. Khảo sát các điểm xác suất thực tế và ngưỡng mặc định. Trường hợp không thể phát hiện dương mà giữ FPR, chấp nhận dự đoán toàn âm và báo F1=0.
7. Chọn model chính trong các bản safe raw/engineered bằng validation AP, rồi F1/Recall/FPR. Lưu `selection_lock.json` **trước** vòng báo cáo test. Tất cả model và threshold đã khóa được đánh giá trong một lượt báo cáo test để cung cấp bảng so sánh; không thay đổi lựa chọn sau khi xem bảng này.

Thí nghiệm khó hơn `profile_generalization.json`: GroupKFold theo profile trên **train+validation** thôi, với HGB có tham số cố định trước và threshold 0.5. Mỗi fold giữ toàn bộ một số profile chưa có trong train fold; đồng thời xe không giao nhau vì mỗi xe có một profile. Đây là thí nghiệm riêng, không dùng model/hyperparameter được chọn theo validation để tránh kết quả CV giả độc lập, và không tác động việc chọn model chính. Chỉ 14 profile và vẫn là giả lập nên không chứng minh tổng quát hóa ra mọi dòng xe.

## 4. Đọc kết quả

`outputs/reports/model_comparison.csv` có các dòng `validation_0.5`, `validation_tuned`, `test_0.5`, `test_tuned`, và `ablation_validation_*`. So sánh đúng stage/threshold, chú ý cờ `engineered` và `selected`. `cv_average_precision` là điểm CV tuning raw trên train, không phải test. AP được tính bằng `average_precision_score`, không phải tích phân trapezoid PR curve; báo cáo dùng tên Average Precision (PR-AUC theo quy ước yêu cầu).

| Metric | Cách hiểu |
| --- | --- |
| Accuracy | Tỷ lệ đúng toàn bộ; dễ cao với model luôn dự đoán âm khi Y2 hiếm |
| Balanced Accuracy | Trung bình Recall của lớp âm và lớp dương |
| Precision | Trong các phiên báo dương, bao nhiêu phiên thật sự dương theo nhãn mô phỏng |
| Recall | Trong các phiên nhãn dương, bao nhiêu phiên được phát hiện |
| F1 | Trung bình điều hòa Precision/Recall; thấp nếu một thành phần thấp |
| Average Precision / PR-AUC | Chất lượng xếp hạng lớp dương trên các ngưỡng; baseline ngẫu nhiên gần tỷ lệ dương |
| ROC-AUC | Khả năng xếp hạng dương trên âm; chỉ tính khi tập có hai lớp |
| FPR | `FP / (FP + TN)`, tỷ lệ phiên âm bị báo dương; giới hạn validation không bảo đảm FPR test/xe thật |
| Confusion matrix | Hàng là nhãn thật, cột là dự đoán, thứ tự `[[TN, FP], [FN, TP]]` |

Threshold quyết định từ xác suất sang 0/1; thay ngưỡng đổi Precision/Recall/FPR nhưng không đổi AP/ROC-AUC. Xác suất mô hình chưa được calibration và chỉ phản ánh mô phỏng; không coi đó là xác suất lỗi xe thật.

`evaluation.json` lưu model được chọn, validation/test ở ngưỡng tuned và 0.5, cùng tỷ lệ nhãn. Các biểu đồ confusion matrix/PR/ROC nằm ở `outputs/plots/`. `*_permutation_importance.csv` đo mức giảm AP khi xáo trộn một đầu vào trên validation, với trung bình và độ lệch chuẩn. Giá trị âm hoặc gần 0 có thể do nhiễu; biến tương quan có thể chia sẻ importance. Importance không chứng minh nhân quả và không được dùng để chọn lại feature theo test.

EDA gồm `data_quality_report.json`, `feature_summary.csv`, `missing_values.csv`, `target_distribution.csv`, `joint_target_distribution.csv`, `targets_by_vehicle_id.csv`, `targets_by_profile_id.csv`, `numeric_correlations.csv`, `data_dictionary.csv` và histogram trong `outputs/eda/plots/`.

## 5. Predict CSV mới

Sau khi train:

```bash
python predict.py --input outputs/reports/example_input.csv --models-dir outputs/models --output outputs/predictions.csv
python predict.py --input du_lieu_moi.csv --models-dir outputs/models --output outputs/du_doan_moi.csv
```

`example_input.csv` chứa 5 quan sát đã có từ validation, bỏ nhãn và biến bị loại; đây là ví dụ inference, không phải dữ liệu thật hoặc thêm dữ liệu train. CSV mới cần đủ 17 cột raw SAFE_FEATURES như bảng trên; không cần nhãn, ID hay tự tính bốn engineered feature. Có thể dùng header và kiểu dữ liệu của file ví dụ làm mẫu. ID nếu có sẽ được giữ trong kết quả để đối chiếu, không dùng làm đầu vào model.

Predict đọc `inference_schema.json`, kiểm tra cột bắt buộc và giá trị numeric/giới hạn vật lý, xác minh hash model khớp metadata, nạp hai joblib pipeline, dùng threshold đã lưu. Một số ô thiếu được cảnh báo và impute bằng thống kê train; nếu thiếu toàn bộ một feature cần thiết thì dừng để yêu cầu số đo/ước lượng hợp lệ. Đầu vào ngoài khoảng quan sát train được cảnh báo vì có nguy cơ ngoại suy. Category mới được encoder xử lý; các cờ vật lý 0/1 vẫn phải hợp lệ.

Output có `<target>_probability`, `<target>_prediction`, `<target>_threshold` cho Y1/Y2 và `interpretation=SIMULATION_RESEARCH_ONLY`. Xác suất phải hữu hạn trong [0,1]. Không ghi đè input hoặc ghi kết quả vào `data/raw/`. Không suy dòng điện từ giá bán màn hình, thể loại nhạc, ca sĩ hoặc dung tích động cơ.

## 6. Cấu trúc và các file chính

```text
khtk/
├── README.md
├── requirements.txt
├── .gitignore
├── huan_luyen_yaris_ml.py     # CLI EDA, tuning, lựa chọn, test, lưu model
├── predict.py                # CLI inference CSV, kiểm tra schema
├── data/
│   ├── raw/                  # Ba file dữ liệu gốc giữ nguyên nội dung
│   └── processed/            # Chỗ dành cho dữ liệu dẫn xuất, hiện chưa sử dụng
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Load, audit CSV/Excel/reference, chia nhóm
│   ├── feature_engineering.py # Whitelist, phạm vi, policy 43 cột, công thức
│   ├── modeling.py           # Pipeline encoding/imputation, model/search space
│   └── evaluation.py         # Metrics, threshold validation, biểu đồ
├── tests/
│   ├── conftest.py
│   ├── test_data_processing.py
│   ├── test_data_leakage.py
│   ├── test_group_split.py
│   └── test_inference.py
└── outputs/                  # Artifacts được sinh ra, bỏ qua trong Git
    ├── eda/
    ├── models/               # y1_pipeline.joblib, y2_pipeline.joblib,
    │                        # thresholds.json, inference_schema.json
    ├── reports/              # Comparison, evaluation, features, config, split,
    │                        # selection lock, tuning, importance, ví dụ CSV
    └── plots/
```

Tất cả đường dẫn mặc định được xây từ `pathlib.Path` và vị trí project, không hardcode máy cá nhân. Giữ thư mục `src/` khi di chuyển artifacts sang máy khác để joblib nạp được custom transformer/wrapper.

## 7. Tests và tái lập

```bash
python -m pytest -q
```

Tests kiểm tra BOM/schema/nhãn, range sai/Inf/chuỗi sai, dư địa điện âm hợp lệ, scenario trùng/ID thiếu/nhãn thiếu, imputation numeric/categorical và unseen hãng/mẫu, thống kê imputer không thay đổi khi predict validation, leakage/target/ID không ảnh hưởng dự đoán, công thức và missing propagation, nhóm split/CV không trùng, reload model, threshold/FPR, lỗi schema inference và xác suất hợp lệ. Test end-to-end thực sự gọi CLI train với Dummy/Logistic rồi predict CSV trong output tạm; không ghi đè các artifacts chính. Test XGBoost bỏ qua nếu không có thư viện.

Seed cố định và phiên bản core trong requirements giúp tái lập trong cùng môi trường; khác CPU/thư viện vẫn có thể có sai khác dấu phẩy động nhỏ. `training_config.json` lưu version thực tế, SHA-256 dữ liệu, seed, search space, split và thời gian chạy.

## 8. Giới hạn nghiên cứu

Không có quan sát training từ xe thật và không có nhãn chẩn đoán thực nghiệm. Việc tách theo xe giúp tránh rò rỉ giữa các phiên của cùng xe giả, nhưng toàn bộ dữ liệu vẫn xuất phát từ cùng mô phỏng; model có thể học quy tắc của bộ sinh dữ liệu. F1/AP cao không xác nhận độ chính xác ngoài đời, tác động nhân quả, công suất máy phát của chiếc Yaris cụ thể hoặc độ an toàn của việc nâng cấp.

Không dùng kết quả để khẳng định cần thay máy phát, kết luận màn hình gây chết máy hay quyết định sửa chữa khi chưa có dữ liệu thực nghiệm. Muốn nghiên cứu áp dụng cần thu thập số đo hợp lệ, xác minh VIN/mã phụ tùng, có nhãn chuyên môn độc lập và đánh giá trên xe thật chưa có trong huấn luyện. Các “sức khỏe %” là proxy cần định nghĩa phép đo rõ ràng; không phải đại lượng tự có chỉ bằng nhập tên mẫu xe.

Tài liệu phương pháp: [scikit-learn: tránh leakage qua Pipeline](https://scikit-learn.org/stable/common_pitfalls.html), [cross-validation theo nhóm](https://scikit-learn.org/stable/modules/cross_validation.html). Nguồn thông số xe được lưu trong CSV reference và sheet `Sources`; pipeline kiểm tra tính nhất quán với dữ liệu đã cung cấp, không tự tuyên bố xác minh các nguồn đó cho xe người dùng.

## 9. Kết quả phiên kiểm tra thực tế

Trước khi nhận yêu cầu dừng chạy thử, pipeline đã chạy với seed 2026 và cấu hình mặc định trên Python 3.13.9: **24 tests đạt, 1 test XGBoost bỏ qua vì thư viện chưa cài**, không có warning trong lần chạy tests cuối. Huấn luyện đầy đủ, lưu/nạp lại hai model, kiểm tra dependency và predict 5 dòng đã thành công. Dữ liệu raw giữ nguyên hash.

| Nhãn | Model được chọn bằng validation | Threshold | Test Accuracy | BAcc | Precision | Recall | F1 | AP | ROC-AUC | FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Y1 | HistGradientBoosting (engineered) | 0.6170 | 0.8773 | 0.8289 | 0.8785 | 0.7015 | 0.7801 | 0.8911 | 0.9355 | 0.0436 |
| Y2 | LogisticRegression (raw) | 0.2576 | 0.9005 | 0.8071 | 0.4107 | 0.6970 | 0.5169 | 0.6357 | 0.9402 | 0.0827 |

Y2 có AP validation 0,9348 nhưng AP test 0,6357, Precision test chỉ 0,4107. F1 test ở ngưỡng tuned 0,5169 thấp hơn ngưỡng 0.5 (0,5758), dù Recall tăng. Giữ nguyên lựa chọn từ validation; không đổi model hoặc threshold theo kết quả test. Với 33 phiên dương trong test và dữ liệu hoàn toàn giả lập, chưa có bằng chứng đủ cho ứng dụng xe thật.

Bảng test của **mọi thuật toán và cả hai biến thể raw/engineered**, confusion matrices, ablation và xác nhận thực thi ở [báo cáo bàn giao](outputs/reports/completion_report.md); dữ liệu máy đọc ở [evaluation.json](outputs/reports/evaluation.json), [model_comparison.csv](outputs/reports/model_comparison.csv) và [verification.json](outputs/reports/verification.json). Đây là snapshot của phiên đã kiểm tra; sau khi chạy lại, xem artifacts mới để biết kết quả hiện hành.
