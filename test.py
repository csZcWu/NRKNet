import os
from time import time

import cv2
import lpips
import numpy as np
import torch
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from tqdm import tqdm
import config as config
from NRKNet import NRKNet
from data_v3 import TestDataset
from utils import load_model, set_requires_grad, compute_psnr

loss_fn_alex = lpips.LPIPS(net='alex').cuda()


def setup_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    setup_seed(0)
    path_to_save = 'result_on_' + config.test['dataset'] + '_trained_on_' + config.train['train_dataset_name']
    os.makedirs(path_to_save, exist_ok=True)
    DPDD_dataset = TestDataset('Dataset_test_' + config.test['dataset'])
    dataloader = DataLoader(DPDD_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0,
                            pin_memory=True)

    nrknet = NRKNet(config).cuda()
    # net = DRBNet_single().cuda()

    total = sum([param.nelement() for param in nrknet.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))
    set_requires_grad(nrknet, False)
    max_psnr = 0
    e = 0
    last_epoch = load_model(nrknet, config.train['resume'], epoch=config.train['resume_epoch'])
    # a = net.GCM.kernels

    log_dir = 'test/{}'.format('DPDD')
    os.system('mkdir -p {}'.format(log_dir))
    psnr_list = []
    ssim_list = []
    lpips_list = []

    total_time = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            tt = time()
            torch.cuda.synchronize()
            st = time()
            dbs, bs = nrknet(batch['img256'].cuda(), None, 'test')
            torch.cuda.synchronize()
            et = time()
            if step:
                total_time += (et - st)
            torch.cuda.synchronize()

            if config.test['dataset'] == 'RTF':
                psnr_list.append(
                    compute_psnr(torch.round(dbs[-1] * 255.)[:, :, 4:364, 4:364],
                                 torch.round(batch['label256'].cuda() * 255.)[:, :, 4:364, 4:364],
                                 255).cpu().numpy())
                ssim_list.append(
                    ssim(torch.round(dbs[-1] * 255.)[:, :, 4:364, 4:364],
                         torch.round(batch['label256'].cuda() * 255.)[:, :, 4:364, 4:364], data_range=255,
                         size_average=True).cpu().numpy())
                lpips_list.append(loss_fn_alex(torch.round(dbs[-1][:, :, 4:364, 4:364] * 255.) / 255. * 2 - 1,
                                               batch['label256'][:, :, 4:364, 4:364].cuda() * 2 - 1).cpu().numpy()[0][
                                      0][0][0])
                cv2.imwrite(path_to_save + '/' + str(step).rjust(4, '0') + '.png',
                            dbs[-1][:, :, 4:364, 4:364].squeeze(0).cpu().numpy().transpose([1, 2, 0]) * 255.)

            else:
                psnr_list.append(
                    compute_psnr(torch.round(dbs[-1] * 255.), torch.round(batch['label256'].cuda() * 255.),
                                 255).cpu().numpy())
                ssim_list.append(
                    ssim(torch.round(dbs[-1] * 255.), torch.round(batch['label256'].cuda() * 255.), data_range=255,
                         size_average=True).cpu().numpy())
                lpips_list.append(loss_fn_alex(torch.round(dbs[-1] * 255.) / 255. * 2 - 1,
                                               batch['label256'].cuda() * 2 - 1).cpu().numpy()[0][0][0][0])
                cv2.imwrite(path_to_save + '/' + str(step).rjust(4, '0') + '.png',
                            dbs[-1].squeeze(0).cpu().numpy().transpose([1, 2, 0]) * 255.)

    psnr = np.mean(psnr_list)
    print()
    print()
    print('psnr:', psnr)
    ssim = np.mean(ssim_list)
    print('ssim:', ssim)
    lpips_ = np.mean(lpips_list)
    print('lpips:', lpips_)
    print('avg_time:', total_time / (len(dataloader) - 1))