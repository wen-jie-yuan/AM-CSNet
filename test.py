# -*-coding:utf-8-*-
# ! /usr/bin/env python
"""
Author And Time : ywj 2021/10/22 20:39
Desc: test
"""
import argparse
import glob
import os

import scipy.io as sio
import torch

from utils import utility
# set the path of test model
from utils.utility import pad_delete, pad_image

model_path = './saved_model/model_25.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=lambda storage, loc: storage)
model = model.to(device)
parser = argparse.ArgumentParser(description='CSNet')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--sampling_rate', type=str, default='25', help='save reconstruct images')
parser.add_argument('--dir_data', type=str, default='./dataset/', help='dataset directory')
parser.add_argument('--dir', type=str, default='./res_images', help='save reconstruct images')
parser.add_argument('--data_test', type=str, default='Set5+Set11+Set14+BSD100', help='test dataset name')
parser.add_argument('--save_results',  type=bool, default='False',  help='save output results')
args = parser.parse_args()
args.data_test = args.data_test.split('+')

with torch.no_grad():
    for dataset in args.data_test:
        image_list = glob.glob(args.dir_data + "/test_images_mat/{}_mat/*.*".format(dataset))
        avg_psnr = 0.0
        avg_ssim = 0.0
        for image_name in image_list:
            image = sio.loadmat(image_name)['im_gt_y']
            image = image.astype(float)

            im_input = image / 255.
            im_input = torch.from_numpy(im_input).float().view(1, -1, im_input.shape[0], im_input.shape[1])

            im_input, row_p, col_p = pad_image(im_input, 32)

            if not args.cpu:
                im_input = im_input.cuda()
            im_output = model(im_input).squeeze()
            im_output = pad_delete(im_output, row_p, col_p)
            image_res = utility.normalize_255(im_output)

            psnr = utility.calc_psnr_255(image, image_res)
            ssim = utility.calc_ssim(image, image_res)
            avg_psnr += psnr
            avg_ssim += ssim

            # if args.save_results:
            #     path = os.path.join(args.dir, args.sampling_rate, 'results')
            #     if not os.path.exists(path):
            #         os.makedirs(path)
            #     utility.save_image(image_res, psnr, ssim, path, dataset, image_name)

        avg_psnr = avg_psnr / len(image_list)
        avg_ssim = avg_ssim / len(image_list)
        print('[{}]\tPSNR: {:.2f}\tSSIM: {:.4f}'.format(dataset, avg_psnr, avg_ssim))
