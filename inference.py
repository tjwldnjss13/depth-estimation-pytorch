import os
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from PIL import Image

from dataset.kitti_dataset import *
from models.dispnet import *
from utils.disparity import *


def get_dataset_loader():
    root = 'C://DeepLearningData/KITTI/'
    dset = KITTIDataset(root, 'train')
    loader = DataLoader(dset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    return loader


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loader = get_dataset_loader()
    model = DispNet().to(device)

    ckpt_pth = './pretrain/Adam_dispnet_kitti_200epoch_0.000100lr_0.24034loss(train)_0.32626loss_0.31909loss(ap)_0.00313loss(ds)_0.00403loss(lr).ckpt'
    ckpt = torch.load(ckpt_pth)
    model.load_state_dict(ckpt['model_state_dict'])

    transform = T.Resize((375, 1242))

    for i, (imgl, imgr) in enumerate(loader):
        imgl, imgr = imgl[0].unsqueeze(0).to(device), imgr[0].unsqueeze(0).to(device)

        disp = model(imgl)
        dr = disp[:, 0].unsqueeze(1)
        dl = disp[:, 1].unsqueeze(1)
        pred_imgr = get_image_from_disparity(imgl, dr)
        pred_imgl = get_image_from_disparity(imgr, dl)

        imgl = transform(imgl)
        imgr = transform(imgr)
        dr = transform(dr)
        dl = transform(dl)
        pred_imgr = transform(pred_imgr)
        pred_imgl = transform(pred_imgl)

        imgl = imgl.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        imgr = imgr.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        dr = dr.squeeze().detach().cpu().numpy()
        dl = dl.squeeze().detach().cpu().numpy()
        pred_imgr = pred_imgr.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        pred_imgl = pred_imgl.squeeze().permute(1, 2, 0).detach().cpu().numpy()

        # base line : .54m, focal length : 721
        depth_dr = .54 * 721 / (1242 * dr)
        depth_dl = .54 * 721 / (1242 * dl)

        plt.subplot(321)
        plt.imshow(imgl)
        plt.subplot(322)
        plt.imshow(imgr)
        plt.subplot(323)
        plt.imshow(dr)
        plt.subplot(324)
        plt.imshow(dl)
        plt.subplot(325)
        plt.imshow(pred_imgl)
        plt.subplot(326)
        plt.imshow(pred_imgr)
        plt.show()


if __name__ == '__main__':
    main()
















