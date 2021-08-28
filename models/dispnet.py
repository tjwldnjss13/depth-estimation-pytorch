import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conv import *
from models.stn import *
from utils.disparity import *


class DispNet(nn.Module):
    def __init__(self, img_size=(384, 768)):
        super().__init__()
        _, self.w = img_size
        self.max_disparity = int(.3 * self.w)
        self.ec = nn.Sequential(
            Conv(3, 64, 7, 2, 3),
            Conv(64, 128, 5, 2, 2),
            nn.Sequential(
                Conv(128, 256, 5, 2, 2),
                Conv(256, 256, 3, 1, 1)
            ),
            nn.Sequential(
                Conv(256, 512, 3, 2, 1),
                Conv(512, 512, 3, 1, 1)
            ),
            nn.Sequential(
                Conv(512, 512, 3, 2, 1),
                Conv(512, 512, 3, 1, 1)
            ),
            nn.Sequential(
                Conv(512, 1024, 3, 2, 1),
                Conv(1024, 1024, 3, 1, 1)
            )
        )
        self.dc = nn.Sequential(
            Upconv(1024, 512, 4, 2, 1, 0),
            Upconv(512, 256, 4, 2, 1, 0),
            Upconv(256, 128, 4, 2, 1, 0),
            Upconv(128, 64, 4, 2, 1, 0),
            Upconv(64, 32, 4, 2, 1, 0)
        )
        self.pred_ec = nn.Conv2d(1024, 2, 1, 1, 0)
        self.pred_dc = nn.Sequential(
            nn.Conv2d(512, 2, 3, 1, 1),
            nn.Conv2d(256, 2, 3, 1, 1),
            nn.Conv2d(128, 2, 3, 1, 1),
            nn.Conv2d(64, 2, 3, 1, 1),
            nn.Conv2d(32, 2, 3, 1, 1)
        )
        self.ic_dc = nn.Sequential(
            nn.Conv2d(512 + 2 + 512, 512, 3, 1, 1),
            nn.Conv2d(256 + 2 + 512, 256, 3, 1, 1),
            nn.Conv2d(128 + 2 + 256, 128, 3, 1, 1),
            nn.Conv2d(64 + 2 + 128, 64, 3, 1, 1),
            nn.Conv2d(32 + 2 + 64, 32, 3, 1, 1)
        )
        self.upconv_last = Upconv(2, 2, 4, 2, 1, 0, use_activation=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)
        self._init_weights()
        self.stn = STN()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, image):
        ec_list = []
        pred_list = []
        ic_list = []

        x = image
        for e in self.ec:
            x = e(x)
            ec_list.append(x)
        pred_list.append(self.pred_ec(x))

        for i in range(len(self.dc)):
            upc = self.dc[i](ec_list.pop(-1)) if i == 0 else self.dc[i](ic_list.pop(-1))
            ic_list.append(self.ic_dc[i](torch.cat([upc, F.interpolate(pred_list.pop(-1), scale_factor=2, mode='bilinear', align_corners=True), ec_list.pop(-1)], dim=1)))
            pred_list.append(self.pred_dc[i](ic_list[-1]))

        # disp = F.interpolate(pred_list[0], scale_factor=2, mode='bilinear', align_corners=True)
        disp = self.upconv_last(pred_list[0])
        disp = self.sigmoid(disp) * .3

        return disp


if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DispNet().to(device)
    summary(model, (3, 384, 768))

















