# import libs
from __future__ import print_function
import matplotlib.pyplot as plt
#% matplotlib inline

import os
import cv
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from scipy import io
import scipy.io as io
import numpy as np
from models import *

import torch
import torch.optim
from PIL import Image
from models.Unet2D import UNet
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize =-1
PLOT = True
sigma = 25
sigma_ = sigma/255.
s1 = 128
s2 = 128

# load noise image
fp = open(r'inter_img.img','rb')
img_noisy_np = np.fromfile(fp,dtype=np.float32).reshape((1, s1, s2))
img_max = img_noisy_np.max()
img_noisy_np = img_noisy_np / img_max
img_noisy_torch = torch.from_numpy(img_noisy_np).unsqueeze(0).type(dtype)

# load 'wx' as the loss function's weight
fp2 = open(r'weight.img','rb')
weight_img = np.fromfile(fp2,dtype=np.float32).reshape((1, s1, s2))
weight_img_max = weight_img.max()
weight_img = weight_img / weight_img_max
weight_img_torch = torch.from_numpy(weight_img).unsqueeze(0).type(dtype)

# Set up
INPUT = 'noise'  # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net'  # 'net,input'

reg_noise_std = 1. / 30.  # set to 1./20. for sigma=50
LR = 1e-2 # 1e-2 to 1e-4

OPTIMIZER = 'adam'  # 'LBFGS'
#OPTIMIZER = 'LBFGS'
exp_weight = 0.99

matr = io.loadmat(r'sub_iteration.mat')
Dip_iter = matr['sub_iteration']
Dip_iter = Dip_iter.max()
num_iter = Dip_iter
show_every = Dip_iter - 1

in_channels = 1
inter_channels = 16
out_channels = 1
net = UNet(in_channels, inter_channels, out_channels)
net = net.type(dtype)


matr1 = io.loadmat(r'sub_iter.mat')
iter = matr1['sub_iter']
iter = iter.max()

# the first sub-iteration in the first epoch of Recon. is randomly selected if iter = 0 or can use better initialization.
if iter != 0:
    f1 = os.path.join(r'trained_model', 'DIP_Unet_{}iter.ckpt'.format(iter-1))
    net.load_state_dict(torch.load(f1))


# load image prior or random noise as nerual network's input: here x-ray CT is our prior!
#net_input = get_noise(1, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
fp1 = open(r'Prior_CT.img','rb')
img_noisy_np1 = np.fromfile(fp1, dtype=np.float32).reshape((1, s1, s2))
img_noisy_np1 = (img_noisy_np1 - img_noisy_np1.min())/ (img_noisy_np1.max() - img_noisy_np1.min())
net_input = torch.from_numpy(img_noisy_np1).unsqueeze(0).type(dtype)

# Compute number of parameters
#s = sum([np.prod(list(p.size())) for p in net.parameters()])
#print('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)
noise = net_input.detach().clone()


i = 0


def closure():
    loss = []
    global i, net_input
    out = net(net_input)

    # loss function is obtained from optimization transfer!!!
    total_loss = 1/2 * mse(torch.sqrt(weight_img_torch) * out, torch.sqrt(weight_img_torch) * img_noisy_torch)

    net.zero_grad()
    total_loss.backward()
    loss.append(total_loss.item())

    if PLOT and i % show_every == 0:
        #print('Iteration %05d    Loss %08f ' % (i, total_loss.item()))
        f = os.path.join(r'trained_model', 'DIP_Unet_{}iter.ckpt'.format(i))
        torch.save(net.state_dict(), f)
        #print('DIP done')
    i += 1

    return total_loss


p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)
out_np = torch_to_np(net(net_input))
denor_out = out_np * img_max
fp3 = open('DIP_output.img','wb')
denor_out.tofile(fp3)
