import torch
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image

from metrics.ssim import *


if __name__ == '__main__':
    img_pth = 'samples/depth_example.png'
    img = Image.open(img_pth)
    img = np.array(img)
    print(img.shape)
    print(np.min(img), np.max(img))
    plt.imshow(img)
    plt.show()


