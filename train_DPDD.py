import sys

import torch.nn.functional as F
from pytorch_msssim import ssim
from tqdm import tqdm

import config as config
from NRKNet import NRKNet
from data_v3 import Dataset, TestDataset
from log import TensorBoardX
from utils import *

# from time import time

log10 = np.log(10)
MAX_DIFF = 1
le = 1
mse = torch.nn.MSELoss().cuda()
mae = torch.nn.L1Loss().cuda()


def SSIM_loss(output, target):
    return 1 - torch.mean(ssim(output, target, data_range=1, size_average=False))


def FFT_loss(output, target):
    target_fft = torch.fft.fft2(target)
    output_fft = torch.fft.fft2(output)
    return mae(target_fft.real, output_fft.real) + mae(target_fft.imag, output_fft.imag)


def compute_loss(dbs, bs, clear256, blurry256, epoch, phase='train'):
    assert dbs[-1].shape[0] == clear256.shape[0]
    scales = len(dbs)
    global le
    le = 1
    if phase == 'train':
        blurrys = [F.interpolate(blurry256, scale_factor=(0.5)**i, mode='bilinear') for i in range(scales-1)]
        clears = [clear256]+[F.interpolate(clear256, scale_factor=(0.5) ** i, mode='bilinear') for i in range(scales - 1)]
        weights = [1 - 0.25 * (scales - i - 1) for i in range(scales)]
        blurrys.reverse()
        clears.reverse()

        temp = (epoch % 1000) % 200
        ssim_loss, reblur_loss, mse_loss, fft_loss = 0, 0, 0, 0

        if temp > 100 and epoch >= 1000:
            le = 10
            loss = 0
            loss += mse(dbs[-1], clear256)
            psnr = 10 * torch.log(MAX_DIFF ** 2 / loss) / log10
            for i in range(len(clears)):
                ssim_loss += SSIM_loss(dbs[i], clears[i]) * weights[i]

        else:
            le = 1
            '''use mse loss'''
            mse_loss = mse(dbs[-1], clear256)
            psnr = 10 * torch.log(MAX_DIFF ** 2 / mse_loss) / log10
            mse_loss = 0
            for i in range(scales):
                mse_loss += mse(dbs[i], clears[i]) * weights[i]
                fft_loss += FFT_loss(dbs[i], clears[i]) * weights[i]
        for i in range(len(bs)):
            reblur_loss += mse(bs[i], blurrys[i]) * weights[i]

        total_loss = mse_loss + ssim_loss + 0.1 * fft_loss
        if epoch < 2000:
            le = 20

        return {'mse': total_loss, 'psnr': psnr}
    else:
        loss = mse(dbs[-1], clear256)
        psnr = 10 * torch.log(MAX_DIFF ** 2 / loss) / log10
        return {'mse': loss, 'psnr': psnr}


def backward(loss, optimizer):
    optimizer.zero_grad()
    with torch.autograd.set_detect_anomaly(True):
        loss['mse'].backward()
    optimizer.step()
    return


def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id


def worker_init_fn_seed1(worker_id):
    seed = 11
    seed += worker_id


def setup_seed(seed):
    seed = int(seed)
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def set_learning_rate(optimizer, epoch):
    # decreasing learning rate
    if 3000 >= epoch >= 2000:
        optimizer.param_groups[0]['lr'] = config.train['learning_rate'] * 0.5
    # fixed learning rate
    elif 4000 >= epoch > 3000:
        optimizer.param_groups[0]['lr'] = config.train['learning_rate'] * 0.25
    elif epoch > 4000:
        optimizer.param_groups[0]['lr'] = config.train['learning_rate'] * 0.1
    else:
        optimizer.param_groups[0]['lr'] = config.train['learning_rate']
    # optimizer.param_groups[0]['lr'] = config.train['learning_rate']


def train(config):
    setup_seed(0)
    tb = TensorBoardX(config_filename='train_config.py', sub_dir=config.train['sub_dir'])
    log_file = open('{}/{}'.format(tb.path, 'train.log'), 'w')
    train_dataset = Dataset('Dataset_train_DPDD'])
    test_dataset = TestDataset('Dataset_test_DPDD'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train['batch_size'],
                                                   shuffle=True, drop_last=True, num_workers=4, pin_memory=True,
                                                   worker_init_fn=worker_init_fn_seed)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.train['test_batch_size'],
                                                  shuffle=False, drop_last=True, num_workers=1, pin_memory=True)

    nrknet = torch.nn.DataParallel(NRKNet(config)).cuda()
    total = sum([param.nelement() for param in nrknet.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))
    assert config.train['optimizer'] in ['Adam', 'SGD']
    if config.train['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(nrknet.parameters(), lr=config.train['learning_rate'],
                                     weight_decay=config.loss['weight_l2_reg'])
    else:
        optimizer = torch.optim.SGD(nrknet.parameters(), lr=config.train['learning_rate'],
                                    weight_decay=config.loss['weight_l2_reg'], momentum=config.train['momentum'],
                                    nesterov=config.train['nesterov'])

    last_epoch = -1

    if config.train['resume'] is not None:
        last_epoch = load_model(nrknet, config.train['resume'], epoch=config.train['resume_epoch'])

    if config.train['resume_optimizer'] is not None:
        _ = load_optimizer(optimizer, nrknet, config.train['resume_optimizer'], epoch=config.train['resume_epoch'])
        assert last_epoch == _

    train_loss_log_list = []
    val_loss_log_list = []
    first_val = True

    best_val_psnr = 0
    best_epoch = None
    print('last epoch', last_epoch)

    for epoch in tqdm(range(last_epoch + 1, config.train['num_pre-train_epochs']), file=sys.stdout):

        if epoch == 3000 or epoch == 4000 or epoch == 2000:
            _ = load_model(nrknet, config.train['resume'], epoch=best_epoch)
            optimizer = torch.optim.Adam(nrknet.parameters(), lr=config.train['learning_rate'],
                                         weight_decay=config.loss['weight_l2_reg'])
        set_learning_rate(optimizer, epoch)
        nrknet.train()
        tb.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch * len(train_dataloader), 'train')
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), file=sys.stdout,
                                desc='training'):
            batch['label256'] = batch['label256'].cuda()
            batch['img256'] = batch['img256'].cuda()
            dbs, bs = nrknet(batch['img256'], batch['label256'], )
            # print(len(dbs))
            loss = compute_loss(dbs, bs, batch['label256'],
                                batch['img256'], epoch, 'train')
            backward(loss, optimizer)

            for k in loss:
                loss[k] = float(loss[k].cpu().detach().numpy())
            train_loss_log_list.append({k: loss[k] for k in loss})
            for k, v in loss.items():
                tb.add_scalar(k, v, epoch * len(train_dataloader) + step, 'train')
        # test and log
        if first_val or epoch % le == le - 1 or epoch % 50 == 0:
            nrknet.eval()
            with torch.no_grad():
                first_val = False
                for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader),
                                        file=sys.stdout,
                                        desc='validating'):
                    tt = time.time()
                    batch['label256'] = batch['label256'].cuda()
                    batch['img256'] = batch['img256'].cuda()
                    dbs, bs = nrknet(batch['img256'], phase='test')
                    loss = compute_loss(dbs, bs, batch['label256'], batch['img256'], epoch, 'test')
                    for k in loss:
                        loss[k] = float(loss[k].cpu().detach().numpy())
                    val_loss_log_list.append({k: loss[k] for k in loss})
                train_loss_log_dict = {k: float(np.mean([dic[k] for dic in train_loss_log_list])) for k in
                                       train_loss_log_list[0]}
                val_loss_log_dict = {k: float(np.mean([dic[k] for dic in val_loss_log_list])) for k in
                                     val_loss_log_list[0]}
                for k, v in val_loss_log_dict.items():
                    tb.add_scalar(k, v, (epoch + 1) * len(train_dataloader), 'val')
                if best_val_psnr < val_loss_log_dict['psnr']:
                    best_epoch = epoch
                    best_val_psnr = val_loss_log_dict['psnr']
                    save_model(nrknet, tb.path, epoch)
                    save_optimizer(optimizer, nrknet, tb.path, epoch)
                train_loss_log_list.clear()
                val_loss_log_list.clear()
                log_msg = ""
                log_msg += "epoch {} ".format(epoch)

                log_msg += " | train : "
                for idx, k_v in enumerate(train_loss_log_dict.items()):
                    k, v = k_v
                    if k == 'acc':
                        log_msg += "{} {:.3%} {}".format(k, v, ',')
                    else:
                        log_msg += "{} {:.5f} {}".format(k, v, ',')
                log_msg += "  | val : "
                for idx, k_v in enumerate(val_loss_log_dict.items()):
                    k, v = k_v
                    if k == 'acc':
                        log_msg += "{} {:.3%} {}".format(k, v, ',')
                    else:
                        log_msg += "{} {:.5f} {}".format(k, v, ',' if idx < len(val_loss_log_list) - 1 else '')
                tqdm.write(log_msg, file=sys.stdout)
                sys.stdout.flush()
                log_file.write(log_msg + '\n')
                log_file.flush()


if __name__ == "__main__":
    setup_seed(0)
    train(config=config)
