import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    def __init__(self, feature_size=(384, 768)):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, 7),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 92 * 188, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        nn.init.zeros_(self.fc_loc[2].weight)
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, image, disparity):
        size = image.size()

        x_temp = self.localization(disparity)
        x_temp = x_temp.view(-1, 10 * 92 * 188)
        theta = self.fc_loc(x_temp)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, size)
        x = F.grid_sample(image, grid)

        return x


if __name__ == '__main__':
    from torchsummary import summary
    stn = STN().cuda()
    x = torch.ones(2, 30, 384, 768).cuda()
    disp = torch.zeros(2, 1, 384, 768).cuda()
    pred = stn(x, disp)
    print(pred)
    print(pred.shape)

















