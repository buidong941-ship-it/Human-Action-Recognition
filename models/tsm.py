import torch.nn as nn

from models.resnet_tsm import make_tsm_resnet50


class TSM_Network(nn.Module):
    def __init__(self, num_classes=51, n_segment=16):
        super().__init__()
        # Khởi tạo backbone ResNet-50 đã được chèn các lớp Temporal Shift
        self.model = make_tsm_resnet50(num_classes, n_segment)

    def forward(self, x):
        # x shape: [Batch, Time, Channel, H, W] -> (Ví dụ: [4, 16, 3, 224, 224])
        b, t, c, h, w = x.shape
        
        # Bước 1: Ép Batch và Time lại để ResNet 2D có thể đọc được
        # New shape: [Batch * Time, Channel, H, W] -> ([64, 3, 224, 224])
        x = x.view(b * t, c, h, w)
        
        # Bước 2: Chạy qua mạng ResNet-50 (đã có TSM trao đổi thông tin giữa các frames)
        logits = self.model(x) # Output shape: [64, 51] (51 là số lớp hành động)
        
        # Bước 3: Tách lại Batch và Time, sau đó lấy trung bình theo chiều Time (dim=1)
        # logits.view(b, t, -1) -> [4, 16, 51]
        # .mean(dim=1) -> [4, 51] (Kết quả cuối cùng cho 4 video)
        return logits.view(b, t, -1).mean(dim=1)
