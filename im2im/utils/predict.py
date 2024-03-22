# =======================================================================
# file name:    predict.py
# description:  generate segmentation mask
# authors:      Xihan Ma, Mingjie Zeng
# date:         2023-02-25
# version:
# =======================================================================
import torch
import torch.nn as nn
from torch import Tensor

import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects, binary_opening, binary_closing, binary_erosion
from skimage.morphology import disk
from skimage.morphology import convex_hull_object
from skimage.filters import median


def morphology_recon(bin_input: np.ndarray):
  bin_output = bin_input[0, :, :].copy()
  # ===== params =====
  small_object_thresh = 0.4
  small_hole_thresh = 0.1

  # ===== perform area opening & closing =====
  binary_opening(image=bin_output, footprint=disk(3), out=bin_output)
  binary_closing(image=bin_output, footprint=disk(3), out=bin_output)

  # ===== reject small objects & fill holes =====
  area = np.sum(bin_input)
  remove_small_objects(ar=bin_output, min_size=int(small_object_thresh*area), out=bin_output)
  remove_small_holes(ar=bin_output, area_threshold=int(small_hole_thresh*area), out=bin_output)

  bin_output = np.expand_dims(bin_output, axis=0)
  return bin_output


def smooth_boundary(bin_input: np.ndarray):
  '''
  '''
  bin_output = bin_input[0, :, :].copy()
  median(image=bin_output, footprint=disk(3), out=bin_output)
  binary_erosion(image=bin_output, footprint=disk(3), out=bin_output)
  bin_output = np.expand_dims(bin_output, axis=0)
  return bin_output


def form_convex_hull(bin_input: np.ndarray):
  '''
  '''
  # ===== find convex hull =====
  bin_output = convex_hull_object(bin_input[0, :, :])
  # print(f'convex hull shape: {bin_output.shape}, max label: {np.max(bin_output)}')
  bin_output = np.expand_dims(bin_output, axis=0)
  return bin_output


def regularize_bin_msk(bin_input: np.ndarray):
  ''' post-processing on binary segmentation mask
  '''
  bin_output = bin_input.copy()
  bin_output = morphology_recon(bin_output)
  # bin_output = form_convex_hull(bin_output)
  # bin_output = smooth_boundary(bin_output)
  return bin_output


def predict(image: Tensor, net: nn.modules, thresh: float = 0.3, enReg = False, device: torch.device = torch.device('cuda')):
  '''
  @param image:
  @param net:
  @param thresh:
  '''
  prob_pred_raw = net(image)
  # print(f'mask requires gradient: {prob_pred_raw.requires_grad}')

  prob_pred = prob_pred_raw.clone().detach()

  for ch in range(net.n_classes):
    # ===== gen probabilistic output =====
    channel = prob_pred_raw[:, ch, :, :]
    # channel[channel > 1.0] = 1.0
    # prob_pred[:, ch, :, :] = channel

  if net.training:
    prob_pred.requires_grad_(requires_grad=True)

  return prob_pred.float()

def att_predict(image: Tensor, net: nn.modules, thresh: float = 0.3, enReg = True, device: torch.device = torch.device('cuda')):
  '''
  @param image:
  @param net:
  @param thresh:
  '''
  mask_pred_raw = net(image)
  # print(f'mask requires gradient: {mask_pred_raw.requires_grad}')
  #print(f"type of psi: {type(psi)}")
  # print(f"shape of psi: {psi.shape}")
  #print(f"type of mask_pred_raw: {type(mask_pred_raw)}")

  mask_pred = mask_pred_raw[0].clone().detach()
  prob_pred = mask_pred_raw[0].clone().detach()
  psi = mask_pred_raw[1].clone().detach()
  # print(f"pred_psi:{psi}")
  # print(f"type of psi: {type(psi)}")
  # print(f"shape of psi: {psi.shape}")

  for ch in range(net.n_classes):
    # ===== gen binary mask =====
    channel_bin = mask_pred_raw[0][:, ch, :, :] > thresh
    if enReg:
      channel_bin = regularize_bin_msk(channel_bin.cpu().numpy())
    mask_pred[:, ch, :, :] = torch.tensor(channel_bin).to(device=device)

    # ===== gen probabilistic mask (AFC map) =====
    channel = mask_pred_raw[0][:, ch, :, :]
    # channel[channel < thresh] = 0
    prob_pred[:, ch, :, :] = channel

  if net.training:
    mask_pred.requires_grad_(requires_grad=True)
    prob_pred.requires_grad_(requires_grad=True)

  return mask_pred.long(), prob_pred.float(), psi
