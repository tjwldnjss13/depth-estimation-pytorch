import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    def __init__(self, in_channels, input_shape):
        super().__init__()
        mid_channels = in_channels * 2
        h_in, w_in = input_shape

        self.out_channels = in_channels * 3
        self.h_out, self.w_out = math.floor((h_in - 14) / 4), math.floor((w_in - 14) / 4)

        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 7),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, self.out_channels, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(self.out_channels * self.h_out * self.w_out, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        nn.init.zeros_(self.fc_loc[2].weight)
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        size = x.size()
        x_temp = x

        x = self.localization(x)
        x = x.view(-1, self.out_channels * self.h_out * self.w_out)
        theta = self.fc_loc(x)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, size)
        x = F.grid_sample(x_temp, grid)

        return x


if __name__ == '__main__':
    from torchsummary import summary
    c, h, w = 128, 96, 192
    stn = STN(c, (h, w)).cuda()
    x = torch.ones(2, c, h, w).cuda()
    summary(stn, (c, h, w))

















