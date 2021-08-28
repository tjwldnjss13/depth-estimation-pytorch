import os
import torch.utils.data as data
import torchvision.transforms as T

from PIL import Image


class KITTIDataset(data.Dataset):
    def __init__(self, root, mode='train'):
        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = T.Compose([T.Resize((384, 768)), T.ToTensor()])
        self.imgl_pth_list, self.imgr_pth_list = self._get_image_path_list()

    def _get_image_path_list(self):
        l_img_pth_list = []
        r_img_pth_list = []
        img_dir = os.path.join(self.root, 'raw', self.mode)
        sub_dirs = os.listdir(img_dir)
        for sub_dir in sub_dirs:
            dir_l = os.path.join(img_dir, sub_dir, 'image_02', 'data')
            dir_r = os.path.join(img_dir, sub_dir, 'image_03', 'data')
            pths_l = os.listdir(dir_l)
            pths_r = os.listdir(dir_r)
            for p_l in pths_l:
                l_img_pth_list.append(os.path.join(dir_l, p_l))
            for p_r in pths_r:
                r_img_pth_list.append(os.path.join(dir_r, p_r))

        return l_img_pth_list, r_img_pth_list

    def __getitem__(self, idx):
        imgl, imgr = self.imgl_pth_list[idx], self.imgr_pth_list[idx]
        imgl, imgr = Image.open(imgl), Image.open(imgr)

        imgl, imgr = self.transform(imgl), self.transform(imgr)

        return imgl, imgr

    def __len__(self):
        return len(self.imgl_pth_list)


def custom_collate_fn(batch):
    item1 = [item[0] for item in batch]
    item2 = [item[1] for item in batch]

    return item1, item2


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    root = 'C://DeepLearningData/KITTI/'
    dset = KITTIDataset(root)
    imgl = Image.open(dset.imgl_pth_list[0])
    imgr = Image.open(dset.imgr_pth_list[0])
    plt.subplot(211)
    plt.imshow(imgl)
    plt.subplot(212)
    plt.imshow(imgr)
    plt.show()
