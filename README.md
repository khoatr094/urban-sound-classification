# Báo cáo phân loại âm thanh đô thị — 7 lớp

## Phạm vi
Mô hình: CNN4 tham khảo Walden, ResCNN tự xây dựng, AST pretrained fine-tune và ensemble.
Chỉ giữ car_horn, children_playing, dog_bark, engine_idling, gun_shot, siren, drilling.
Loại air_conditioner, jackhammer, street_music. Không gộp jackhammer vào drilling.

## Phương pháp và giới hạn so sánh
Holdout/partial: test folds [10]. Chế độ: full. Train/validation/test cùng dữ liệu cho mọi mô hình; purge nguồn trùng ưu tiên test.
Nếu mode=smoke, mọi kết quả chỉ là kiểm thử kỹ thuật, không dùng để viết kết luận hiệu năng.
Chuẩn hóa CNN fit trên train; AST dùng chuẩn hóa pretrained AudioSet. Augmentation chỉ train.
CNN4 sử dụng 4 khối 32/64/128/256, kernels (3,5),(3,5),(5,5),(5,5), điều chỉnh mono 16 kHz và pooling toàn cục.
Bài Walden dùng stereo 44.1 kHz, tập 7 lớp có điều hòa thay cho drilling, chia 80/20 và báo cáo 97.82%.
Đây không phải tái lập chính xác bài báo. Không so trực tiếp accuracy giữa hai thiết lập.
Nguồn pretraining AST: AudioSet; chưa kiểm toán trùng nguồn giữa AudioSet và UrbanSound8K.
Các fold được purge không còn là benchmark UrbanSound8K nguyên bản. Xem split.json và purged.csv từng fold.
Chọn checkpoint, ensemble và mô hình demo bằng validation macro-F1. Test dùng báo cáo so sánh sau khi chọn.
Thời gian eval gồm đọc đặc trưng và truyền dữ liệu, không phải độ trễ realtime end-to-end.
Unknown dựa ngưỡng heuristic, cần bộ dữ liệu ngoài lớp để hiệu chỉnh và đánh giá thực tế.

## Tệp kết quả
- outputs/data: audit, metadata, thống kê, manifest train/val/test, nguồn trùng và mẫu bị purge.
- outputs/checkpoints: best.pt, last.pt, history.csv, validation predictions và trạng thái hoàn tất.
- outputs/metrics: accuracy, precision, recall, F1, confusion, ROC/AP, bảng so sánh và trọng số ensemble.
- outputs/figures: PNG/SVG để chèn báo cáo; outputs/predictions: xác suất từng mẫu và phân tích lỗi.
- Vẽ lại biểu đồ từ CSV đã lưu: `.\.venv\Scripts\python.exe scripts\plot_results.py`.
  Script tự tìm các fold, tạo biểu đồ train/validation loss, validation accuracy và
  macro-F1 riêng cho từng mô hình, đường so sánh theo epoch, cùng biểu đồ so sánh
  accuracy/precision/recall/F1, tốc độ suy luận (clip/giây), kích thước checkpoint,
  số tham số, VRAM và thời gian train. Biểu đồ được lưu tại `outputs/figures/foldNN/` và
  thêm vào `outputs/report.html` nếu báo cáo này đã tồn tại. Có thể chỉ định fold,
  ví dụ `.\.venv\Scripts\python.exe scripts\plot_results.py --fold 10`.
- configs: cấu hình, tên lớp, normalization, AST config/processor và nguồn pretrained.
- src/engine.py, scripts/predict.py, scripts/plot_results.py: mã và lệnh dự đoán/vẽ lại sau tải về.
- notebooks: snapshot mã đã chạy; notebook có đầy đủ output được lưu bằng Kaggle Save Version.
Cache đặc trưng phục vụ chạy tiếp, không đưa vào ZIP để tránh nhân đôi dung lượng.

## Tài liệu tham khảo
1. https://arxiv.org/abs/2209.04075
2. https://urbansounddataset.weebly.com/urbansound8k.html
3. https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer
4. https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593

## Chạy ứng dụng Gradio trên Windows

Ứng dụng tìm checkpoint theo đường dẫn tương đối với thư mục dự án. Hỗ trợ
checkpoint fold dạng `outputs/checkpoints/fold10/<model>/best.pt` và các
checkpoint dạng phẳng `outputs/checkpoints/<model>_best.pt` đang có trong gói này.
- AST processor/config, nhãn và chuẩn hóa CNN được đọc tại chỗ từ `configs/`.
- Không cần tải hoặc đóng gói lại WAV của UrbanSound8K. Notebook snapshot được giữ ở
  `notebooks/executed_code_snapshot.ipynb`.

1. Cài một phiên bản Python tương thích với bản PyTorch dùng trên máy và cài
   PyTorch tương thích CPU hoặc CUDA **trong môi trường Python hệ thống** trước.
   Ứng dụng và kiểm thử trong workspace này đã chạy với Python 3.14.8.
   `run_windows.bat` tạo `.venv` với
   `--system-site-packages`, kiểm tra PyTorch đã có và không cài/cập nhật PyTorch,
   CUDA, torchvision hay torchaudio.
2. Mở thư mục dự án trong Explorer, nhấp đúp `run_windows.bat`, hoặc trong
   PowerShell chạy `.\run_windows.bat`.
3. Mở URL localhost được in ra. Chọn/ghi âm tệp, mô hình và CPU/GPU, sau đó bấm
   **Dự đoán**. Microphone chỉ ghi âm rồi dự đoán, không phải nhận dạng trực tiếp.

Ứng dụng cung cấp bốn lựa chọn với các checkpoint phẳng hiện có: CNN4, ResCNN,
AST và ensemble. CNN4/ResCNN được nạp bằng kiến trúc tương ứng với tên và kích
thước trọng số trong checkpoint. Nếu có `outputs/metrics/fold10/ensemble.json`,
ensemble dùng trọng số ở đó; nếu không có, ensemble demo lấy trung bình xác suất
của các model có checkpoint. Ensemble demo không có trọng số huấn luyện đi kèm
không phải ensemble dùng để tạo các số liệu trong báo cáo.
Model/checkpoint được giữ trong bộ nhớ đến khi tắt ứng dụng; các lần sau không
phải nạp lại checkpoint.
Thời gian hiển thị trong thông tin kỹ thuật có cả model/tiền xử lý và tổng thời
gian xử lý phía app. Tốc độ thực tế phụ thuộc thiết bị, độ dài và định dạng âm
thanh; không thể đảm bảo mọi máy đều dưới 3 giây.

Nếu không có PyTorch tương thích trong Python hệ thống, script sẽ dừng và hướng
dẫn cài PyTorch bằng lệnh phù hợp với máy từ trang chọn cài đặt chính thức của
PyTorch; không tự thay PyTorch/CUDA hiện có. Chỉ thêm package phụ bằng
`python -m pip install -r requirements.txt`.

### Thêm checkpoint khác

Giữ nguyên cấu trúc thư mục do Output Kaggle cung cấp, ví dụ
`outputs/checkpoints/fold10/ast/best.pt`; checkpoint dạng phẳng như
`outputs/checkpoints/ast_best.pt`, `cnn4_best.pt` và `rescnn_best.pt` cũng được
hỗ trợ. Với AST, cần có thêm
`configs/ast_7class/config.json`, `configs/ast_processor/preprocessor_config.json`,
`configs/stats_fold10.json` và cấu hình trong `configs/config.json`. Khi không
có `ensemble.json`, app tạo ensemble demo bằng trung bình đều xác suất; cách
này không tương đương trọng số ensemble được chọn trên validation trong báo cáo.
Ứng dụng không tải model thay thế từ internet.

### Thiết bị, tệp âm thanh và ngưỡng từ chối

Ứng dụng dùng CUDA khi được chọn và CUDA khả dụng, nếu không có thể chọn CPU.
Chỉ các codec WAV, FLAC, OGG, MP3 mà `soundfile`/libsndfile tại máy giải mã được
mới sử dụng được; nếu MP3 không giải mã được, hãy cài codec cần thiết hoặc chuyển
tệp sang WAV. Âm thanh ngắn hơn 4 giây được đệm cho đủ cửa sổ suy luận; các tệp
rỗng, hỏng, chứa NaN/Inf, im lặng hoặc dài quá 900 giây (15 phút) bị từ
chối với thông báo tiếng Việt. Waveform hiển thị toàn bộ thời lượng tệp; có thể
dùng nút biểu tượng trong “Thông tin kỹ thuật” để mở/ẩn phổ cửa sổ đầu 4 giây.
Waveform, Log-Mel và các phím tắt đều nằm trong phần này. Đổi
giao diện sáng/tối bằng biểu tượng ở đầu app; lựa chọn được lưu trong trình duyệt.
Nút toàn màn hình nằm cạnh biểu tượng giao diện; nhấn Escape để thoát.
Ctrl+O chọn tệp, Ctrl+M bật/tắt micro, Ctrl+Enter dự đoán. Lịch sử hiển thị ở cột bên
phải trên màn hình rộng, hiển thị danh sách cao hơn và chuyển xuống dưới trên màn hình nhỏ. Biểu đồ waveform
và Log-Mel được tăng chiều cao để dễ quan sát. Xác suất 7 lớp được sắp xếp từ cao
xuống thấp; tên tệp dự đoán hiển thị dưới kết quả.

`demo_threshold` trong `configs/config.json` (Output hiện tại: 0.6) chỉ là ngưỡng
từ chối tham khảo. Ngưỡng này chưa được đánh giá bằng bộ âm thanh ngoài 7 lớp;
kết quả “Chưa chắc chắn” không khẳng định âm thanh nằm ngoài 7 lớp. Xác suất là
softmax của mô hình, không phải xác suất hiệu chuẩn.

### Kiểm thử và kết quả fold 10

Chạy `.\.venv\Scripts\python.exe -m pytest tests`. Bộ kiểm thử kiểm tra checkpoint/config, WAV hợp lệ,
âm thanh im lặng, tệp rỗng/định dạng không hỗ trợ và suy luận AST có đủ 7 xác suất
hữu hạn tổng xấp xỉ 1. Kiểm thử suy luận tải checkpoint AST thật và có thể cần
nhiều RAM/thời gian.

Tab đánh giá của app đọc `outputs/metrics/fold10/test_comparison.csv` nếu tệp
này tồn tại; nếu chưa có, app hiển thị thông báo thiếu số liệu. Số liệu được
ghi nhãn rõ là test fold 10 của lần chạy hiện tại, không phải độ chính xác tổng quát. Tệp âm thanh
demo được notebook tham chiếu là
`/kaggle/input/datasets/f8kinwd/drilling/drilling-into-concrete-32298.mp3`;
đường dẫn Kaggle này không nằm trong Output đã tải về, nên cần cung cấp lại file
này nếu muốn kiểm tra mẫu tiếng khoan đó trên máy Windows.
