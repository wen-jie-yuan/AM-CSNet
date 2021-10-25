import datetime
import math
import os
import sys
import time
import cv2
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from skimage import measure

matplotlib.use('Agg')

TOTAL_BAR_LENGTH = 80
LAST_T = time.time()
BEGIN_T = LAST_T


def progress_bar(current, total, msg=None):
    global LAST_T, BEGIN_T
    if current == 0:
        BEGIN_T = time.time()  # Reset for new bar.

    current_len = int(TOTAL_BAR_LENGTH * (current + 1) / total)
    rest_len = int(TOTAL_BAR_LENGTH - current_len) - 1

    sys.stdout.write(' %d/%d' % (current + 1, total))
    sys.stdout.write(' [')
    for i in range(current_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    current_time = time.time()
    step_time = current_time - LAST_T
    LAST_T = current_time
    total_time = current_time - BEGIN_T

    time_used = '  Step: %s' % format_time(step_time)
    time_used += ' | Tot: %s' % format_time(total_time)
    if msg:
        time_used += ' | ' + msg

    msg = time_used
    sys.stdout.write(msg)

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


# return the formatted time
def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    seconds_final = int(seconds)
    seconds = seconds - seconds_final
    millis = int(seconds * 1000)

    output = ''
    time_index = 1
    if days > 0:
        output += str(days) + 'D'
        time_index += 1
    if hours > 0 and time_index <= 2:
        output += str(hours) + 'h'
        time_index += 1
    if minutes > 0 and time_index <= 2:
        output += str(minutes) + 'm'
        time_index += 1
    if seconds_final > 0 and time_index <= 2:
        output += str(seconds_final) + 's'
        time_index += 1
    if millis > 0 and time_index <= 2:
        output += str(millis) + 'ms'
        time_index += 1
    if output == '':
        output = '0ms'
    return output


def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, win, data_range=1023, size_average=True, full=False):
    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ssim(X, Y, win_size=11, win_sigma=10, win=None, data_range=1, size_average=True, full=False):
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=False,
                         full=True)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ms_ssim(X, Y, win_size=11, win_sigma=10, win=None, data_range=1, size_average=True, full=False, weights=None):
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if weights is None:
        weights = torch.FloatTensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(X.device, dtype=X.dtype)

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim(X, Y,
                             win=win,
                             data_range=data_range,
                             size_average=False,
                             full=True)
        mcs.append(cs)

        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)
    # weights, (level)
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1))
                            * (ssim_val ** weights[-1]), dim=0)  # (batch, )

    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3):
        super(SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range

    def forward(self, X, Y):
        return ssim(X, Y, win=self.win, data_range=self.data_range, size_average=self.size_average)


class MS_SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3, weights=None):
        super(MS_SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights

    def forward(self, X, Y):
        return ms_ssim(X, Y, win=self.win, size_average=self.size_average, data_range=self.data_range,
                       weights=self.weights)


class Timer:
    def __init__(self):
        self.acc = 0
        self.t0 = time.time()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart:
            self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


class CheckPoint:
    def __init__(self, args):
        self.ok = True
        self.args = args
        self.psnr_log = torch.Tensor()

        # set checkPoint.dir
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if not args.load:
            if not args.save:
                args.save = args.model + '-' + str(args.sub_rate) + '-' + now
            self.dir = os.path.join('experiment', args.save)
        else:
            self.dir = os.path.join('experiment', args.load)
            if os.path.exists(self.dir):
                self.psnr_log = torch.load(self.get_path('psnr_log.pt'))
                if not args.test:
                    print('Continue from epoch {}...'.format(len(self.psnr_log)))
                else:
                    print('Testing model from epoch {}...'.format(int(args.resume)))

        # make dirs
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)

        for d in args.data_test:
            os.makedirs(self.get_path('results', d), exist_ok=True)
            os.makedirs(self.get_path('results_2', d), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)

        # write option information into config.txt
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    # save model, optimizer, loss, psnr and plot loss, psnr
    def save(self, trainer, epoch, is_best=False):
        self.plot_psnr(epoch)
        trainer.loss.plot_loss(self.dir, epoch)

        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        torch.save(trainer.optimizer.state_dict(), os.path.join(self.dir, 'optimizer.pt'))
        torch.save(self.psnr_log, self.get_path('psnr_log.pt'))

    def save2(self, trainer, epoch, is_best=False):
        self.plot_psnr(epoch)
        trainer.loss_model1.plot_loss(self.dir, epoch)
        trainer.loss_model2.plot_loss(self.dir, epoch)

        trainer.model1.save(self.dir, epoch, is_best=is_best)
        trainer.model2.save(self.dir, epoch, is_best=is_best)
        trainer.loss_model1.save(self.dir)
        trainer.loss_model2.save(self.dir)
        torch.save(trainer.optimizer_model1.state_dict(), os.path.join(self.dir, 'optimizer_model1.pt'))
        torch.save(trainer.optimizer_model2.state_dict(), os.path.join(self.dir, 'optimizer_model2.pt'))
        torch.save(self.psnr_log, self.get_path('psnr_log.pt'))

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def add_psnr_log(self, psnr_log):
        self.psnr_log = torch.cat([self.psnr_log, psnr_log])

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'psnr on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            plt.plot(
                axis,
                self.psnr_log[:, idx_data].numpy(),
                label='sub rate {}'.format(self.args.sub_rate)
            )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    # save rec images
    def save_results(self, dataloader, filename, rec_image, psnr):
        filename = self.get_path('results', dataloader.dataset.name, '{}-{}.png'.format(filename[0], str(psnr)))
        normalized = rec_image[0].mul(255 / self.args.rgb_range)
        tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
        imageio.imwrite(filename, tensor_cpu.numpy())


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def normalize_255(img):
    image_res = img.cpu().numpy().astype(np.float32)
    image_res = image_res * 255.
    image_res[image_res < 0] = 0
    image_res[image_res > 255.] = 255.
    return image_res


def calc_psnr(img1, img2, rgb_range):
    diff = (img1 - img2) / rgb_range

    if diff.size(1) > 1:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        diff = diff.mul(convert).sum(dim=1)

    mse = diff.pow(2).mean()
    return -10 * math.log10(mse)


def calc_psnr_255(img1, img2):
    diff = img1 - img2
    rmse = math.sqrt(np.mean(diff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def calc_ssim(img1, img2):
    img1 = np.array(img1, dtype=np.uint8)
    img2 = np.array(img2, dtype=np.uint8)
    if img1.shape[-1] == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (score, diff) = measure.compare_ssim(img1, img2, full=True)

    return score


def save_image(image, psnr, ssim, path, dataset, filename):
    path = os.path.join(path, dataset)
    filename, _ = os.path.splitext(os.path.basename(filename))
    filename = '{}-{:.2f}-{:.4f}.png'.format(filename, psnr, ssim)

    image = image.astype(np.uint8)
    imageio.imwrite(os.path.join(path, filename), image)


def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    kwargs = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs['betas'] = args.betas
        kwargs['eps'] = args.epsilon
    else:
        optimizer_function = optim.Adam

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    scheduler = None

    if args.decay_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )

    elif args.decay_type == 'multi':
        milestones = list(map(lambda x: int(x), args.lr_multi.split('-')))
        scheduler = optim.lr_scheduler.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler
