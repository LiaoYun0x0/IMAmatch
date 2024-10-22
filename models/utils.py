import torch
from math import log
from loguru import logger

import torch
from einops import repeat
from kornia.utils import create_meshgrid
import torch.nn.functional as F
import random
import numpy as np

##############  ↓  Coarse-Level supervision  ↓  ##############

@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long()

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
        (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    w_kpts0_long = w_kpts0.long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    )  # (N, L)
    consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2
    valid_mask = nonzero_mask * covisible_mask * consistent_mask

    return valid_mask, w_kpts0

@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt

@torch.no_grad()
def pre_process_data(data,scale=8):
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    
    scale0 = data['scale0'][:,None] if 'scale0' in data else torch.ones((N,1,1),device=device)
    scale1 = data['scale1'][:,None] if 'scale1' in data else torch.ones((N,1,1),device=device)
    offsets0 = data['offsets0'][:,None] if 'offsets0' in data else torch.ones((N,1,1),device=device)
    offsets1 = data['offsets1'][:,None] if 'offsets1' in data else torch.zeros((N,1,1),device=device)
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt0_i = (scale * grid_pt0_c + 3.5) * scale0 + offsets0
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)
    grid_pt1_i = (scale * grid_pt1_c + 3.5) * scale1 + offsets1

    if 'mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])


    _, w_pt0_i = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
    _, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
    w_pt0_c = ((w_pt0_i - offsets1) / scale1) / scale
    w_pt1_c = ((w_pt1_i - offsets0) / scale0) / scale

    # check if mutual nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
    w_pt1_c_round = w_pt1_c[:, :, :].long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    # drop matches out of boundary
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
    correct_0to1 = loop_back == torch.arange(h0*w0, device=device)[None].repeat(N, 1)
    correct_0to1[:, 0] = False  # ignore the top-left corner

    # construct a gt conf_matrix
    conf_matrix_gt = torch.zeros(N, h0*w0, h1*w1, device=device)
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]

    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.update({'conf_matrix_gt': conf_matrix_gt})
    
    # prepare fine level matches
    if len(b_ids) > 0:
        offsets0 = offsets0[b_ids,0]
        offsets1 = offsets1[b_ids,0]
        scale0 = scale0[b_ids,0]
        scale1 = scale1[b_ids,0]
        
        coords0_d8 = torch.stack([i_ids % w0, torch.div(i_ids, w0,rounding_mode='trunc')],dim=1)
        coords1_d8 = torch.stack([j_ids % w0, torch.div(j_ids, w0,rounding_mode='trunc')],dim=1)
        
        lt_off = torch.randint_like(coords0_d8, low=0,high=4) 
        coords0_d2 = coords0_d8 * 4 + lt_off # randomly select one as query
        coords0_d2_i = (coords0_d2) * 2 * scale0 + offsets0
        coords1_d2 = coords1_d8 * 4 + 1.5 # select the center token as refer
        coords1_d2_i = (coords1_d2) * 2 * scale1 + offsets1
        
        
        offsets_d2 = []
        w_coords0_d2_i_list = []
        for i in range(N):
            m = (b_ids == i)
            _, w_coords0_d2_i = warp_kpts(coords0_d2_i[m][None], data['depth0'][[i]], data['depth1'][[i]], data['T_0to1'][[i]], data['K0'][[i]], data['K1'][[i]])
            offsets_d2.append( (w_coords0_d2_i[0] - coords1_d2_i[m]))
            w_coords0_d2_i_list.append(w_coords0_d2_i[0])
        offsets_d2 = torch.cat(offsets_d2, dim=0) / scale1 / 2
        w_coords0_d2_i = torch.cat(w_coords0_d2_i_list,dim=0)
        
        
        # coords1_d2_step2 = (coords1_d2 + offsets_d2).round()
        coords1_d2_step2 = ((w_coords0_d2_i - offsets1) / scale1 / 2).round() + torch.randint_like(coords0_d8, low=-1,high=2)
        coords1_d2_step2_i = (coords1_d2_step2) * 2 * scale1 + offsets1
        offsets_d2_step2 = (w_coords0_d2_i - coords1_d2_step2_i) / scale1 / 2
        
        mask2 = ((coords1_d2_step2 >= 0) & (coords1_d2_step2 < w0*4)).all(dim=-1)
        if torch.sum(mask2) < mask2.shape[0]:
            coords0_d2 = coords0_d2[mask2]
            coords1_d2 = coords1_d2[mask2]
            offsets_d2 = offsets_d2[mask2]
            coords1_d2_step2 = coords1_d2_step2[mask2]
            offsets_d2_step2 = offsets_d2_step2[mask2]
            b_ids_d2 = b_ids_d2[mask2]
            
        
        if len(b_ids) > 0:
            data.update({
                'b_ids_d2': b_ids,
                'coords0_d2': coords0_d2,
                'coords1_d2': coords1_d2,
                'offsets_d2': offsets_d2,
                'coords1_d2_step2': coords1_d2_step2,
                'offsets_d2_step2': offsets_d2_step2
            })

@torch.no_grad()
def mean_blur_tensor(tensor_image,k=3):
    kernel = torch.ones((tensor_image.shape[1],1,k,k)) / (k**2)
    kernel = kernel.to(tensor_image.device)
    tensor_image = torch.nn.functional.conv2d(input=tensor_image, weight=kernel,stride=(1,1),padding=(k//2,k//2),groups=tensor_image.shape[1])
    return tensor_image


