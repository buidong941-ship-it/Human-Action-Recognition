import torch
import torch.nn as nn
import torchvision.models as models

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=16, n_div=8):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment # Số lượng frames trong 1 video (của bạn là 16)
        self.n_div = n_div # Tỉ lệ channel dùng để dịch chuyển (thường là 1/8)

    def forward(self, x):
        # x có shape: [Batch * Time, Channel, H, W]
        bt, c, h, w = x.size()
        t = self.n_segment
        b = bt // t
        x = x.view(b, t, c, h, w) # Tách ra thành [B, T, C, H, W]

        fold = c // self.n_div
        out = torch.zeros_like(x)
        
        # Cơ chế Shift: Dịch khung hình về trước và sau để trao đổi thông tin
        out[:, :-1, :fold] = x[:, 1:, :fold]         # Shift sang trái (Past)
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold] # Shift sang phải (Future)
        out[:, :, 2*fold:] = x[:, :, 2*fold:]        # Giữ nguyên phần còn lại
        
        return self.net(out.view(bt, c, h, w))

def make_tsm_resnet50(num_classes, n_segment=16):
    # Dùng ResNet-50 pre-trained để học nhanh hơn
    base_model = models.resnet50(weights='IMAGENET1K_V1')
    
    # "Phẫu thuật" chèn TSM vào các khối Bottleneck của ResNet
    def add_tsm(model):
        for name, child in model.named_children():
            if isinstance(child, models.resnet.Bottleneck):
                child.conv1 = TemporalShift(child.conv1, n_segment=n_segment)
            else:
                add_tsm(child)
    
    add_tsm(base_model)
    
    # Thay đổi lớp phân loại cuối cùng cho 51 lớp của HMDB51
    base_model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(base_model.fc.in_features, num_classes)
    )
    return base_model
