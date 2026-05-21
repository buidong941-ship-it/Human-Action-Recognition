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
Nếu sử dụng Kaggle có thể Add Input dataset dùng để train tại URL: 
`https://www.kaggle.com/datasets/midzid/hmdb51`

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
/kaggle/working/Video_Action_Recognition
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

- `final-app-2.ipynb` hiện là demo cho TSM ResNet50
- Nếu muốn chạy notebook/app local, cần chỉnh các path Kaggle hardcode.
- File `tsm_resnet50_best.pt` cần có cùng kiến trúc với `TSM_Network`.
- Thứ tự `LABELS` trong app cần khớp với thứ tự class khi train.
- Nếu gặp lỗi thiếu `torch`, cần cài PyTorch đúng phiên bản phù hợp với CUDA hoặc CPU.

## Cách sử dụng final-app-2.ipynb
- Mở Kaggle Notebook hoặc Jupyter Notebook và upload/mở file `final-app-2.ipynb`.
- Nếu chạy trên Kaggle, thêm model weight `tsm_resnet50_best.pt` vào phần **Input** của notebook. Sau đó kiểm tra biến `path_tsm` trong notebook có trỏ đúng tới file weight hay chưa.
- Chạy cell đầu tiên để clone source code app từ GitHub:

```bash
rm -rf /kaggle/working/CS231_Video_Action_Recognition
git -c credential.helper='' clone --depth 1 \
  https://github.com/buidong941-ship-it/Human-Action-Recognition.git \
  /kaggle/working/Video_Action_Recognition
```

- Chạy cell cài đặt thư viện để cài `gradio`, `decord`, `torchvision`.
- Chạy cell import và load model. Nếu cell này báo lỗi đường dẫn, hãy kiểm tra lại:

```python
sys.path.append('/kaggle/working/Video_Action_Recognition')
path_tsm = "/kaggle/input/models/midzid/tsm-resnet50-best/pytorch/default/1/tsm_resnet50_best.pt"
```

- Chạy các cell định nghĩa hàm `process_video_rgb`, `extract_frames_for_display`, `predict_tsm`.
- Chạy cell tạo giao diện Gradio, sau đó chạy cell `demo.launch(...)`.
- Khi Gradio hiện link public, mở link đó, upload video `.avi` hoặc `.mp4`, bấm **Preprocess** để xem 16 frame đầu vào, rồi bấm **Dự đoán bằng TSM** để xem nhãn hành động và độ tin cậy.
- Có thể dùng các video mẫu trong thư mục `testing/` để kiểm tra nhanh.

Nhận xét: notebook hiện phù hợp để demo inference TSM ResNet50 trên Kaggle. Điểm cần chú ý nhất là `path_tsm` phải đúng với vị trí weight trong Kaggle Input, và thư mục được thêm vào `sys.path` phải chứa file định nghĩa `TSM_Network`.
