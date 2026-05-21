# TSM ResNet50 HAR - CS231

Project nhận diện hành động trong video bằng mô hình **Temporal Shift Module (TSM) + ResNet50** trên bộ dữ liệu **HMDB51**. Repo gồm pipeline huấn luyện theo cấu trúc OOP, notebook huấn luyện gốc và notebook demo app Gradio để dự đoán hành động từ video upload.

## Mục Tiêu

- Huấn luyện mô hình phân loại 51 lớp hành động của HMDB51.
- Sử dụng ResNet50 pretrained ImageNet làm backbone.
- Chèn Temporal Shift Module vào các block ResNet để học thông tin theo thời gian.
- Xây dựng app demo bằng Gradio cho inference video.

## Cấu Trúc Project

```text
.
├── config.py                  # Cấu hình đường dẫn, batch size, số frame, epoch, learning rate
├── train.py                   # Pipeline huấn luyện OOP với class TSMTrainer
├── engine.py                  # Class TrainOneEpoch và Evaluator
├── utils.py                   # Hàm hỗ trợ visualize và vẽ biểu đồ training
├── tsm_resnet50_best.pt       # File weight tốt nhất đã train
├── TSM_ResNet50.ipynb         # Notebook training gốc
├── final-app-2.ipynb          # Notebook Gradio app demo inference
├── data/
│   ├── dataset.py             # HMDB51Dataset và collate_fn
│   └── tranforms.py           # VideoTransform cho train/val
├── models/
│   ├── resnet_tsm.py          # TemporalShift và hàm tạo ResNet50-TSM
│   └── tsm.py                 # Class TSM_Network
└── testing/                   # Một số video dùng để test app
```

## Mô Hình

Input của mô hình có dạng:

```text
[Batch, Time, Channel, Height, Width]
```

Với cấu hình mặc định:

```text
Batch size: 4
Num frames: 16
Image size: 224x224
Num classes: 51
Backbone: ResNet50
Optimizer: Adam
Scheduler: CosineAnnealingLR
Loss: CrossEntropyLoss
```

`TSM_Network` reshape video từ `[B, T, C, H, W]` thành `[B*T, C, H, W]`, đưa qua ResNet50 đã chèn TSM, sau đó lấy trung bình logits theo chiều thời gian để tạo dự đoán cuối cùng cho video.

## Cài Đặt

Khuyến nghị dùng môi trường Python có GPU nếu muốn train.

```bash
pip install torch torchvision tqdm pillow numpy matplotlib opencv-python gradio
```

Nếu chạy trên Kaggle, phần lớn thư viện đã có sẵn. Notebook app có cell cài thêm:

```bash
pip install gradio decord torchvision
```

## Dữ Liệu

Pipeline train mặc định dùng đường dẫn trong `config.py`:

```python
DATA_ROOT = '/kaggle/input/datasets/midzid/hmdb51/hmdb51_data'
```

Cấu trúc dữ liệu mong đợi:

```text
hmdb51_data/
├── brush_hair/
│   ├── video_1/
│   │   ├── frame_0001.jpg
│   │   └── ...
│   └── ...
├── cartwheel/
└── ...
```

Nếu chạy local, hãy sửa `DATA_ROOT` trong `config.py` thành đường dẫn dataset trên máy của bạn.

## Train Mô Hình

Chạy:

```bash
python train.py
```

Pipeline sẽ:

1. Tạo train/val/test dataset.
2. Tạo DataLoader.
3. Khởi tạo `TSM_Network`.
4. Huấn luyện trong `EPOCHS`.
5. Lưu model tốt nhất vào:

```text
tsm_resnet50_best.pt
```

6. Đánh giá trên test set.
7. Vẽ biểu đồ loss/accuracy bằng `plot_history`.

## Chạy App Demo

Mở notebook:

```text
final-app-2.ipynb
```

Notebook này chạy app Gradio để upload video và dự đoán hành động bằng TSM ResNet50.

Trên Kaggle, notebook đang dùng các path:

```python
/kaggle/working/CS231_Video_Action_Recognition
/kaggle/input/models/midzid/tsm-resnet50-best/pytorch/default/1/tsm_resnet50_best.pt
```

Nếu chạy local, cần đổi path model sang:

```python
path_tsm = "tsm_resnet50_best.pt"
```

App sẽ:

- Lấy 16 frame đều từ video.
- Resize về `224x224`.
- Normalize theo ImageNet mean/std.
- Dự đoán class hành động.
- Hiển thị confidence và 16 frame đầu vào.

## Video Test

Thư mục `testing/` có một số video mẫu:

```text
Biking.avi
Diving.avi
flic flac.avi
(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi
```

Bạn có thể upload các video này trong app Gradio để kiểm tra nhanh.

## Lưu Ý

- `final-app-2.ipynb` hiện là demo cho TSM ResNet50, chưa phải app đa mô hình hoàn chỉnh.
- Nếu muốn chạy notebook/app local, cần chỉnh các path Kaggle hardcode.
- File `tsm_resnet50_best.pt` cần có cùng kiến trúc với `TSM_Network`.
- Thứ tự `LABELS` trong app cần khớp với thứ tự class khi train.
- Nếu gặp lỗi thiếu `torch`, cần cài PyTorch đúng phiên bản phù hợp với CUDA hoặc CPU.

## Tác Giả

Project phục vụ môn **Nhập môn Thị giác Máy tính - CS231**.
