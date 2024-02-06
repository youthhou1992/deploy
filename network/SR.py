import torch
from torch import nn 
from torch.nn.functional import interpolate
import torch.onnx


class SuperResolutionNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x, upscale_factor):
        # x = interpolate(x, scale_factor=upscale_factor.item(), mode='bicubic', align_corners=False)  #导出onnx模型无法记录torch.Tensor.item()操作
        x = NewInterpolate.apply(x, upscale_factor)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out

class NewInterpolate(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, scales): #定义映射到ONNX算子的行为
        # print("scales in symbolic: ", scales)
        return g.op("Resize", input, g.op("Constant", value_t=torch.tensor([], dtype=torch.float32)),
        scales, coordinate_transformation_mode_s="pytorch_half_pixel",
        cubic_coeff_a_f=-0.75, mode_s='cubic', nearest_mode_s='floor')

    @staticmethod
    def forward(ctx, input, scales):
        scales = scales.tolist()[2:]
        print("scales:", scales)
        return interpolate(input, scale_factor=scales, mode='bicubic', align_corners=False)