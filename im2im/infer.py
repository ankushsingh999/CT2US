# =======================================================================
# file name:    infer.py
# description:  run inference
# authors:      Xihan Ma, Mingjie Zeng
# date:         2023-02-25
# version:
# =======================================================================

import cv2
import time
import torch
import numpy as np
from unet.model import UNet
import torch.nn.functional as F
from utils.vis import array2tensor, tensor2array


# load network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")
net = UNet(n_channels=1, n_classes=1, bilinear=False)
net.to(device=device)
net.load_state_dict(torch.load('checkpoints/checkpoint_epoch20.pth'))
# print(net.eval())

# load example image
image = cv2.imread('../dataset/ct_slices_trans_06-Mar-2024/ct_msk/ctmsk_100.png', cv2.IMREAD_GRAYSCALE)
image = array2tensor(image, device=device)
print(f'input shape: {image.shape}')

for _ in range(100):  # infer on same image 100 times to benchmark fps
  start = time.perf_counter()

  pred = net(image)
  us_sim = tensor2array(pred)*255
  print(f'prediction shape: {pred.shape}')

  print(f'time elapsed: {(time.perf_counter()-start):.3f} sec')  # benchmarking

cv2.imwrite('test_us_sim_out.png', us_sim)
