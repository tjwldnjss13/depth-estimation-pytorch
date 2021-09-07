import os
import time
import datetime
import argparse
import torch
import torch.optim as optim
import torchvision.transforms as tv
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset.kitti_dataset import *
from models.net_custom import *
from metrics.loss import *
from utils.pytorch_util import *
from utils.util import *


def get_kitti_dataset():
    dset_name = 'kitti'
    root = 'D://DeepLearningData/KITTI/'
    dset_train = KITTIDataset(root, 'train')
    dset_val = KITTIDataset(root, 'val')
    collate_fn = custom_collate_fn

    return dset_name, dset_train, dset_val, collate_fn


def adjust_learning_rate(optimizer, current_epoch):
    if isinstance(optimizer, optim.Adam):
        if current_epoch == 0:
            optimizer.param_groups[0]['lr'] = .0001
        elif current_epoch == 5:
            optimizer.param_groups[0]['lr'] = .0001
        elif current_epoch == 20:
            optimizer.param_groups[0]['lr'] = .0001
    elif isinstance(optimizer, optim.SGD):
        if current_epoch == 0:
            optimizer.param_groups[0]['lr'] = .001
        elif current_epoch == 30:
            optimizer.param_groups[0]['lr'] = .001
        elif current_epoch == 40:
            optimizer.param_groups[0]['lr'] = .001


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    now = datetime.datetime.now()
    date_str = f'{now.date().year}{now.date().month:02d}{now.date().day:02d}'
    time_str = f'{now.time().hour:02d}{now.time().minute:02d}{now.time().second:02d}'
    record = open(f'./records/record_{date_str}_{time_str}.csv', 'w')
    record.write('optimizer, epoch, lr, loss(train), loss, loss(ap), loss(ds), loss(lr)\n')
    record.close()

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', required=False, type=int, default=4)
    parser.add_argument('--lr', required=False, type=float, default=.0001)
    parser.add_argument('--momentum', required=False, type=float, default=.9)
    parser.add_argument('--weight_decay', required=False, type=float, default=.0005)
    parser.add_argument('--epoch', required=False, type=int, default=50)

    args = parser.parse_args()
    batch_size = args.batch_size
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    epoch = args.epoch

    dset_name, dset_train, dset_val, collate_fn = get_kitti_dataset()
    train_loader = DataLoader(dataset=dset_train, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(dataset=dset_val, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn, num_workers=2)

    model = DepthNet().to(device)
    model_name = 'depthnet'

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    if isinstance(optimizer, optim.Adam):
        optim_name = 'Adam'
    elif isinstance(optimizer, optim.SGD):
        optim_name = 'SGD'
    ckpt_pth = None
    # ckpt_pth = './pretrain/sigmoid x 0.3/pretrain_Adam_dispnet2_kitti_10epoch_0.000100lr_1.85431loss(train)_2.02557loss_1.92166loss(ap)_0.01510loss(ds)_0.08881loss(lr).ckpt'
    if ckpt_pth is not None:
        ckpt = torch.load(ckpt_pth)
        model.load_state_dict(ckpt['model_state_dict'])
        if optim_name == 'Adam' and 'optimizer_state_dict' in ckpt.keys():
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    train_loss_list = []
    val_loss_list = []

    t_start = time.time()
    for e in range(epoch):
        num_batch = 0
        num_data = 0
        train_loss = 0
        train_loss_ap = 0
        train_loss_ds = 0
        train_loss_lr = 0

        adjust_learning_rate(optimizer, e)
        cur_lr = optimizer.param_groups[0]['lr']
        model.train()
        for i, (imgl, imgr) in enumerate(train_loader):
            num_batch += 1
            num_data += len(imgl)

            print(f'{optim_name} ', end='')
            print(f'[{e+1}/{epoch}] ', end='')
            print(f'{num_data}/{len(dset_train)}  ', end='')
            print(f'<lr> {cur_lr:.6f}  ', end='')

            imgl = make_batch(imgl).to(device)
            imgr = make_batch(imgr).to(device)

            disp = model(imgl)
            # pred_imgr = predict_right_image_from_left_image_and_right_disparity(imgl, dr).to(device)
            optimizer.zero_grad()
            loss, loss_ap, loss_ds, loss_lr = depthnet_loss(imgl, imgr, disp)
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().cpu().item()
            train_loss_ap += loss_ap
            train_loss_ds += loss_ds
            train_loss_lr += loss_lr

            t_batch_end = time.time()
            h, m, s = time_calculator(t_batch_end - t_start)

            print(f'<loss> {loss.detach().cpu().item():.5f} ({train_loss/num_batch:.5f})  ', end='')
            print(f'<loss_ap> {loss_ap:.5f} ({train_loss_ap/num_batch:.5f})  ', end='')
            print(f'<loss_ds> {loss_ds:.5f} ({train_loss_ds/num_batch:.5f})  ', end='')
            print(f'<loss_lr> {loss_lr:.5f} ({train_loss_lr/num_batch:.5f})  ', end='')
            print(f'<time> {int(h):02d}:{int(m):02d}:{int(s):02d}')

            del imgl, imgr, disp, loss, loss_ap, loss_ds, loss_lr
            torch.cuda.empty_cache()

        train_loss /= num_batch
        train_loss_ap /= num_batch
        train_loss_ds /= num_batch
        train_loss_lr /= num_batch

        train_loss_list.append(train_loss)

        num_batch = 0
        val_loss = 0
        val_loss_ap = 0
        val_loss_ds = 0
        val_loss_lr = 0
        model.eval()
        for i, (imgl, imgr) in enumerate(val_loader):
            num_batch += 1

            imgl = make_batch(imgl).to(device)
            imgr = make_batch(imgr).to(device)

            disp = model(imgl)
            loss, loss_ap, loss_ds, loss_lr = depthnet_loss(imgl, imgr, disp)

            val_loss += loss.detach().cpu().item()
            val_loss_ap += loss_ap
            val_loss_ds += loss_ds
            val_loss_lr += loss_lr

            del imgl, imgr, disp, loss, loss_ap, loss_ds, loss_lr
            torch.cuda.empty_cache()

        val_loss /= num_batch
        val_loss_ap /= num_batch
        val_loss_ds /= num_batch
        val_loss_lr /= num_batch

        val_loss_list.append(val_loss)

        record = open(f'./records/record_{date_str}_{time_str}.csv', 'a')
        record_str = f'{optim_name}, {e+1}, {cur_lr}, {train_loss:.5f}, {val_loss:.5f}, {val_loss_ap:.5f}, {val_loss_ds:.5f}, {val_loss_lr:.5f}\n'
        record.write(record_str)
        record.close()

        if (e + 1) % 5 == 0 or e == 0:
            save_pth = f'./save/{optim_name}_{model_name}_{dset_name}_{e+1}epoch_{lr:.6f}lr_{train_loss:.5f}loss(train)_{val_loss:.5f}loss_{val_loss_ap:.5f}loss(ap)_{val_loss_ds:.5f}loss(ds)_{val_loss_lr:.5f}loss(lr).ckpt'
            save_dict = {'model_state_dict': model.state_dict()}
            if optim_name == 'Adam':
                save_dict['optimizer_state_dict'] = optimizer.state_dict()
            torch.save(save_dict, save_pth)

    x_axis = [i for i in range(len(train_loss_list))]
    plt.plot(x_axis, train_loss_list, 'r-', label='train')
    plt.plot(x_axis, val_loss_list, 'b-', label='val')
    plt.title('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()










































