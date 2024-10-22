import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.utils.grid import create_meshgrid
from einops.einops import rearrange


from .matcher.attention_model import MBFormer_248_standardPE,AttentionBlock
from .utils import pre_process_data

EPSILON = 1e-6


class Matcher(nn.Module):
    def __init__(self,config):
        super().__init__()
        dims = config['dims']
        self.backbone = MBFormer_248_standardPE(**config)
        self.window_sizes = {"step1":7, 'step2':3}
        self.attn1 =  AttentionBlock(
                            dims[0],
                            d_head=config['d_spatial'],
                            dropout=config['dropout'],
                            mlp_ratio=config['mbconv_expansion_rate'][-1],
                        )
        self.attn2 =  AttentionBlock(
                        dims[0],
                        d_head=config['d_spatial'],
                        dropout=config['dropout'],
                        mlp_ratio=config['mbconv_expansion_rate'][-1],
                    )
        
        self.regression1 = self.build_linear_pred(2*dims[0] + self.window_sizes["step1"]**2)
        self.regression2 = self.build_linear_pred(2*dims[0] + self.window_sizes["step2"]**2)
    
    def build_linear_pred(self,input_dim,mlp_ratio=2):
        hidden_dim = int(input_dim * mlp_ratio)
        net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh()
        )
        return net

    def compute_coarse_loss(self,conf,conf_gt,weight):
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = 1.0, 1.0
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.
        conf = torch.clamp(conf, 1e-6, 1-1e-6)
        alpha = 0.25
        gamma = 2.0
        pos_conf = conf[pos_mask]
        loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
        if weight is not None:
            loss_pos = loss_pos * weight[pos_mask]
    
        loss = c_pos_w * loss_pos.mean()
        return loss

    def forward(self,data):
        return self.forward_train(data)
    
    def forward_train(self,data):
        features = self.backbone(torch.cat([data['image0'],data['image1']],dim=0)) # [f_d2, f_d4, f_d8]
        device=features[0].device
        features = {
            "2" : features[0],
            "4" : features[1],
            "8" : features[2]
        }
        pre_process_data(data,max_match=1e6)
        
        # compute coarse loss
        feature_c = rearrange(features['8'],'n c h w -> n (h w) c')
        feature_c0, feature_c1 = torch.chunk(feature_c, chunks=2,dim=0)
        feature_c0_norm, feature_c1_norm = map(lambda feat: feat / feat.shape[-1]**.5,
                               [feature_c0, feature_c1])
        similarity_matrix = torch.einsum('nld,nsd->nls',feature_c0_norm, feature_c1_norm) / 0.1
        if 'mask0' in data:
            weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float().to(device)
            similarity_matrix.masked_fill_(
                        ~weight.bool(),
                        -1e9)
        else:
            weight=None
        cf = torch.softmax(similarity_matrix, dim=1) * torch.softmax(similarity_matrix, dim=2)
        loss_coarse = self.compute_coarse_loss(cf, data['conf_matrix_gt'], weight)
        
        
        '''fine steps'''
        loss_fine = torch.zeros_like(loss_coarse)
        if 'b_ids_d2' in data:
            feature0_down, feature1_down = torch.chunk(features['2'],chunks=2,dim=0)
            b,c,h,w = feature0_down.shape
            b_ids = data['b_ids_d2']
            coords0 = data['coords0_d2']
            coords1 = data['coords1_d2']
            offsets = data['offsets_d2']
            coords1_step2 = data['coords1_d2_step2']
            offsets_step2 = data['offsets_d2_step2']
            
            W1 = self.window_sizes['step1']
            W2 = self.window_sizes['step2']
            radius1 = W1 / 2
            radius2 = W2 / 2
            
            window_coords1 = create_meshgrid(W1, W1,False, device).reshape(-1,2) - int(radius1)
            window_coords2 = create_meshgrid(W2, W2,False, device).reshape(-1,2) - int(radius2)
            coords0_windows = coords0[:,None,:] + window_coords1[None,:,:] 
            coords1_windows = coords1[:,None,:] + window_coords1[None,:,:] # (n, w**2, 2)
            coords0_windows_step2 = coords0[:,None,:] + window_coords2[None,:,:]
            coords1_windows_step2 = coords1_step2[:,None,:] + window_coords2[None,:,:]

            # normalize the sample indices to (-1,1)
            coords0_windows[...,0] = coords0_windows[...,0] / (w-1) * 2 - 1
            coords0_windows[...,1] = coords0_windows[...,1] / (h-1) * 2 - 1
            coords1_windows[...,0] = coords1_windows[...,0] / (w-1) * 2 - 1
            coords1_windows[...,1] = coords1_windows[...,1] / (h-1) * 2 - 1
            coords0_windows_step2[...,0] = coords0_windows_step2[...,0] / (w-1) * 2 - 1
            coords0_windows_step2[...,1] = coords0_windows_step2[...,1] / (h-1) * 2 - 1
            coords1_windows_step2[...,0] = coords1_windows_step2[...,0] / (w-1) * 2 - 1
            coords1_windows_step2[...,1] = coords1_windows_step2[...,1] / (h-1) * 2 - 1
            tokens0 = []
            tokens1 = []
            tokens0_step2 = []
            tokens1_step2 = []
            for i in range(b):
                mask = b_ids == i
                if torch.sum(mask) == 0:
                    continue
                tokens0.append(F.grid_sample(feature0_down[[i]], coords0_windows[mask][None],padding_mode='zeros',align_corners=True)[0])
                tokens1.append(F.grid_sample(feature1_down[[i]], coords1_windows[mask][None],padding_mode='zeros',align_corners=True)[0])
                tokens0_step2.append(F.grid_sample(feature0_down[[i]], coords0_windows_step2[mask][None],padding_mode='zeros',align_corners=True)[0])
                tokens1_step2.append(F.grid_sample(feature1_down[[i]], coords1_windows_step2[mask][None],padding_mode='zeros',align_corners=True)[0])
            tokens0 = torch.cat(tokens0, dim=1).permute(1,2,0)
            tokens1 = torch.cat(tokens1, dim=1).permute(1,2,0)
            tokens0_step2 = torch.cat(tokens0_step2,dim=1).permute(1,2,0)
            tokens1_step2 = torch.cat(tokens1_step2,dim=1).permute(1,2,0)
            
            tokens0, tokens1 = self.attn1(tokens0, tokens1)
            tokens0_step2, tokens1_step2 = self.attn2(tokens0_step2, tokens1_step2)
            
            mid_index = W1**2 // 2
            sm = torch.einsum('nlc,nmc->nlm',tokens0, tokens1) / c ** .5
            token = torch.cat([tokens0[:,mid_index,:],tokens1[:,mid_index,:],sm[:,mid_index,:]],dim=-1)
            offsets_pred = self.regression1(token)
            offsets_gt = offsets / radius1
            loss_offsets = ((offsets_gt - offsets_pred) ** 2).sum(-1)

            # compute cls loss. 
            gt_coords = (offsets + int(radius1)).round().clamp(0,W1-1)
            gt_index = (gt_coords[:,0] + gt_coords[:,1] * W1).long()
            cm = torch.softmax(sm, dim=1) * torch.softmax(sm, dim=2)
            select_cm = cm[torch.arange(cm.shape[0]),torch.ones_like(gt_index)*mid_index,gt_index]
            loss_cls = -0.1 * (1-select_cm) * torch.log(select_cm+1e-6)
            mask_step1 = ((offsets_gt >= -1) & (offsets_gt <= 1)).all(dim=-1)
            
            if torch.sum(mask_step1) > 0:
                loss_fine = loss_fine + loss_offsets[mask_step1].mean() + loss_cls[mask_step1].mean()
            else:
                loss_fine = loss_fine + loss_offsets.sum() * 0
            
            # compute stage2 
            mid_index_step2 = W2**2 // 2
            sm_step2 = torch.einsum('nc,nlc->nl',tokens0_step2[:,mid_index_step2], tokens1_step2) / c ** .5
            token_step2 = torch.cat([tokens0_step2[:,mid_index_step2,:],tokens1_step2[:,mid_index_step2,:],sm_step2],dim=-1)
            offsets_pred_step2 = self.regression2(token_step2)
            offsets_gt_step2 = offsets_step2 / radius2
            loss_offsets_step2 = ((offsets_gt_step2 - offsets_pred_step2) ** 2).sum(-1)
            mask_step2 = ((offsets_gt_step2 >= -1) & (offsets_gt_step2 <= 1)).all(dim=-1)
            if torch.sum(mask_step2) > 0:
                loss_fine = loss_fine + loss_offsets_step2[mask_step2].mean()
            else:
                loss_fine = loss_fine + loss_offsets_step2.sum() * 0
        else:
            print('no fine-stage matches, generate some fake matches to avoid hang')
            W1 = self.window_sizes['step1']
            W2 = self.window_sizes['step2']
            radius1 = W1 / 2
            radius2 = W2 / 2
            feature0_down, feature1_down = torch.chunk(features['2'],chunks=2,dim=0)
            n,c,h,w = feature0_down.shape
            tokens0 = feature0_down[:, :, 10:10+W1, 10:10+W1].reshape(n,c,-1).permute(0,2,1).contiguous()
            tokens1 = feature1_down[:, :, 10:10+W1, 10:10+W1].reshape(n,c,-1).permute(0,2,1).contiguous()
            tokens0_step2 = feature0_down[:, :, 11:11+W2, 11:11+W2].reshape(n,c,-1).permute(0,2,1).contiguous()
            tokens1_step2 = feature1_down[:, :, 11:11+W2, 11:11+W2].reshape(n,c,-1).permute(0,2,1).contiguous()
            
            tokens0, tokens1 = self.attn1(tokens0, tokens1)
            tokens0_step2, tokens1_step2 = self.attn2(tokens0_step2, tokens1_step2)
            mid_index = W1**2 // 2
            sm = torch.einsum('nlc,nmc->nlm',tokens0, tokens1) / c ** .5
            token = torch.cat([tokens0[:,mid_index,:],tokens1[:,mid_index,:],sm[:,mid_index,:]],dim=-1)
            offsets_pred = self.regression1(token)
            loss_offsets = ((offsets_pred) ** 2).sum(-1)
            
            mid_index_step2 = W2**2 // 2
            sm_step2 = torch.einsum('nc,nlc->nl',tokens0_step2[:,mid_index_step2], tokens1_step2) / c ** .5
            token_step2 = torch.cat([tokens0_step2[:,mid_index_step2,:],tokens1_step2[:,mid_index_step2,:],sm_step2],dim=-1)
            offsets_pred_step2 = self.regression2(token_step2)
            loss_offsets_step2 = ((offsets_pred_step2) ** 2).sum(-1)
            
            loss_fine = (loss_offsets.sum() + loss_offsets_step2.sum()) * 0
        return loss_coarse, loss_fine
    
    
    def forward_test(self,data, thresh=0.2, high_thresh=0.2,allow_reverse=False,add_matches=0,iter_optimize=False,rplus=1): 
        query,refer = data['image0'], data['image1']
        device = query.device
        b = query.shape[0]
        assert b == 1, 'only support batchsize 1'
        features = self.backbone(torch.cat([query,refer],dim=0)) # [f_d2, f_d4, f_d8]
        features = {
            "2" : features[0],
            "4" : features[1],
            "8" : features[2]
        }
        feature0,feature1 = features['8'].split(b)
        
        b,d,h,w = feature0.shape
        feature0 = rearrange(feature0, 'b c h w -> b (h w) c')
        feature1 = rearrange(feature1, 'b c h w -> b (h w) c')
        
        feature0_norm, feature1_norm = map(lambda feat: feat / feat.shape[-1]**.5,[feature0, feature1])
        sm = torch.einsum('nld,nsd->nls',feature0_norm,feature1_norm) / 0.1
        # if 'mask0' in data:
        #     weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float().to(device)
        #     sm.masked_fill_(~weight.bool(),-1e9)

        cm = torch.softmax(sm,dim=1) * torch.softmax(sm,dim=2)
        mask = cm > thresh
        h_mask = (cm == torch.max(cm, dim=1,keepdim=True)[0])
        v_mask = (cm == torch.max(cm, dim=2,keepdim=True)[0])
        mask_hv = mask * h_mask * v_mask
        
        bindex,qindex,rindex = torch.where(mask_hv)
        if len(bindex) == 0:
            return torch.zeros((0,2),device=bindex.device),torch.zeros((0,2),device=bindex.device),False
        
        coords0 = torch.stack([qindex % w, torch.div(qindex,w,rounding_mode='trunc')],dim=-1).float()
        coords1 = torch.stack([rindex % w, torch.div(rindex,w,rounding_mode='trunc')],dim=-1).float()
        
        reverse = False
        if allow_reverse:
            _,FM = cv2.findFundamentalMat(coords0.cpu().numpy(),coords1.cpu().numpy(), \
                method=cv2.RANSAC,ransacReprojThreshold=3,confidence=0.9999,maxIters=10)
            if FM is not None:
                FM = FM.squeeze()
                if np.sum(FM) > 0:
                    c0 = coords0[FM]
                    c1 = coords1[FM]
                    d0 = (torch.pow(c0[:,None,:] - c0[None,:,:],2).sum(-1)).mean()
                    d1 = (torch.pow(c1[:,None,:] - c1[None,:,:],2).sum(-1)).mean()
                    if d0 < d1:
                        reverse = True
                        mask = (mask * v_mask).transpose(1,2) # allow 1toN matches
                        select_conf = cm.transpose(1,2)[mask]
                    else:
                        mask = mask * h_mask
                        select_conf = cm[mask]
                    bindex,qindex,rindex = torch.where(mask)
                    coords0 = torch.stack([qindex % w, torch.div(qindex,w,rounding_mode='trunc')],dim=-1)
                    coords1 = torch.stack([rindex % w, torch.div(rindex,w,rounding_mode='trunc')],dim=-1)
                else:
                    return torch.zeros((0,2),device=bindex.device),torch.zeros((0,2),device=bindex.device),reverse
            else:
                return torch.zeros((0,2),device=bindex.device),torch.zeros((0,2),device=bindex.device),reverse
        else:
            mask = mask_hv
            select_conf = cm[mask_hv]

        
        if not reverse:
            feature0_down, feature1_down = torch.chunk(features['2'],chunks=2,dim=0)
        else:
            feature1_down, feature0_down = torch.chunk(features['2'],chunks=2,dim=0)
        
        b,c,h,w = feature0_down.shape
        coords0 = (coords0 * 4).float()
        coords1 = (coords1 * 4).float() + rplus
        W1 = self.window_sizes['step1']
        W2 = self.window_sizes['step2']
        radius1 = W1 / 2
        radius2 = W2 / 2
        window_coords1 = create_meshgrid(W1, W1,False, device).reshape(-1,2) - int(radius1)
        window_coords2 = create_meshgrid(W2, W2,False, device).reshape(-1,2) - int(radius2)
        
        if add_matches > 1:
            # add matches for uncertain matches
            conf_m = select_conf < high_thresh
            coords0_high_score = coords0[~conf_m] + 1
            coords1_high_score = coords1[~conf_m]
            conf_high_score = select_conf[~conf_m]
            coords0 = coords0[conf_m]
            coords1 = coords1[conf_m]
            select_conf = select_conf[conf_m]
            assert add_matches in [0,4,9,16], 'unsupported add_matches value {}, please choose from [0,4,9,16]'.format(add_matches)
            if add_matches == 0:
                coords0 = coords0 + 1
            elif add_matches == 4:
                offsets = create_meshgrid(2,2,False, device).flatten(start_dim=1,end_dim=2)+1
                coords0 = (coords0[:,None,:] + offsets).view(-1,2)
                coords1 = (coords1[:,None,:] + offsets * 0).view(-1,2)
                select_conf = torch.tile(select_conf[:,None],(1,4)).view(-1)
            elif add_matches == 9:
                offsets = create_meshgrid(3,3,False, device).flatten(start_dim=1,end_dim=2)
                coords0 = (coords0[:,None,:] + offsets).view(-1,2)
                coords1 = (coords1[:,None,:] + offsets * 0).view(-1,2)
                select_conf = torch.tile(select_conf[:,None],(1,9)).view(-1)
            elif add_matches == 16:
                offsets = create_meshgrid(4,4,False, device).flatten(start_dim=1,end_dim=2)
                coords0 = (coords0[:,None,:] + offsets).view(-1,2)
                coords1 = (coords1[:,None,:] + offsets * 0).view(-1,2)
                select_conf = torch.tile(select_conf[:,None],(1,16)).view(-1)
            coords0 = torch.cat([coords0, coords0_high_score],dim=0)
            coords1 = torch.cat([coords1, coords1_high_score],dim=0)
            select_conf = torch.cat([select_conf, conf_high_score],dim=0)
        else:
            coords0 = coords0 + 1

        # 1st optimize
        coords0_windows = coords0[:,None,:] + window_coords1[None,:,:] 
        coords1_windows = coords1[:,None,:] + window_coords1[None,:,:] # (n, w**2, 2)
        coords0_windows[...,0] = coords0_windows[...,0] / (w-1) * 2 - 1
        coords0_windows[...,1] = coords0_windows[...,1] / (h-1) * 2 - 1
        coords1_windows[...,0] = coords1_windows[...,0] / (w-1) * 2 - 1
        coords1_windows[...,1] = coords1_windows[...,1] / (h-1) * 2 - 1
        tokens0 = F.grid_sample(feature0_down, coords0_windows[None],padding_mode='zeros',align_corners=True)
        tokens0 = tokens0[0].permute(1,2,0)
        tokens1 = F.grid_sample(feature1_down, coords1_windows[None],padding_mode='zeros',align_corners=True)
        tokens1 = tokens1[0].permute(1,2,0)
        # split matches to several chunks to avoid out of memory.
        offsets_pred_list = []
        sm_list = []
        ts = len(tokens0)
        bs = 50000
        chunks = math.ceil(ts / bs)
        for i in range(chunks):
            _tokens0, _tokens1 = self.attn1(tokens0[i*bs:min(ts,(i+1)*bs)], tokens1[i*bs:min(ts,(i+1)*bs)])
            mid_index = W1**2 // 2
            sm = torch.einsum('nlc,nmc->nlm',_tokens0, _tokens1) / c ** .5
            sm_list.append(sm)
            _token = torch.cat([_tokens0[:,mid_index,:],_tokens1[:,mid_index,:],sm[:,mid_index,:]],dim=-1)
            offsets_pred = self.regression1(_token)
            offsets_pred_list.append(offsets_pred)
        sm = torch.cat(sm_list,dim=0)
        offsets_pred = torch.cat(offsets_pred_list,dim=0) * radius1
        coords1 = coords1 + offsets_pred 
        
        if add_matches > 1:
            cm = torch.softmax(sm, dim=1) * torch.softmax(sm, dim=2)
            select_cm = cm[:,mid_index,mid_index] > 0.2
            coords0 = coords0[select_cm]
            coords1 = coords1[select_cm]
            select_conf = select_conf[select_cm]

        # 2nd optimize
        coords1 = coords1.round()
        coords0_windows_step2 = coords0[:,None,:] + window_coords2[None,:,:]
        coords1_windows_step2 = coords1[:,None,:] + window_coords2[None,:,:]
        coords0_windows_step2[...,0] = coords0_windows_step2[...,0] / (w-1) * 2 - 1
        coords0_windows_step2[...,1] = coords0_windows_step2[...,1] / (h-1) * 2 - 1
        coords1_windows_step2[...,0] = coords1_windows_step2[...,0] / (w-1) * 2 - 1
        coords1_windows_step2[...,1] = coords1_windows_step2[...,1] / (h-1) * 2 - 1
        tokens0_step2 = F.grid_sample(feature0_down, coords0_windows_step2[None],padding_mode='zeros',align_corners=True)
        tokens1_step2 = F.grid_sample(feature1_down, coords1_windows_step2[None],padding_mode='zeros',align_corners=True)
        tokens0_step2 = tokens0_step2[0].permute(1,2,0)
        tokens1_step2 = tokens1_step2[0].permute(1,2,0)
        tokens0_step3 = torch.detach_copy(tokens0_step2)
        tokens0_step2, tokens1_step2 = self.attn2(tokens0_step2, tokens1_step2)
        mid_index_step2 = W2**2 // 2
        sm_step2 = torch.einsum('nc,nlc->nl',tokens0_step2[:,mid_index_step2], tokens1_step2) / c ** .5
        token_step2 = torch.cat([tokens0_step2[:,mid_index_step2,:],tokens1_step2[:,mid_index_step2,:],sm_step2],dim=-1)
        offsets_pred_step2 = self.regression2(token_step2)
        coords1 = coords1 + offsets_pred_step2 * radius2
        
        if iter_optimize:
            count_coords1 = coords1
            iter_step = 5
            for i in range(iter_step):
                coords_round = (count_coords1 / (i+1)).round()
                coords1_windows_step3 = coords_round[:,None,:] + window_coords2[None,:,:]
                coords1_windows_step3[...,0] = coords1_windows_step3[...,0] / (w-1) * 2 - 1
                coords1_windows_step3[...,1] = coords1_windows_step3[...,1] / (h-1) * 2 - 1
                tokens1_step3 = F.grid_sample(feature1_down, coords1_windows_step3[None],padding_mode='zeros',align_corners=True)
                tokens1_step3 = tokens1_step3[0].permute(1,2,0)
                _tokens0_step3, _tokens1_step3 = self.attn2(tokens0_step3, tokens1_step3)
                sm_step3 = torch.einsum('nc,nlc->nl',_tokens0_step3[:,mid_index_step2], _tokens1_step3) / c ** .5
                token_step3 = torch.cat([_tokens0_step3[:,mid_index_step2,:],_tokens1_step3[:,mid_index_step2,:],sm_step3],dim=-1)
                offsets_pred_step3 = self.regression2(token_step3)
                coords1 = coords_round + offsets_pred_step3 * radius2
                count_coords1 = count_coords1 + coords1
            coords1 = count_coords1 / (iter_step + 1)
        
        # drop matches out of feature map
        m0 = (coords1 >= 0).all(dim=-1)
        m1 = (coords1[:,0] <= w) & (coords1[:,1] <= h)
        valid_m = m0 & m1
        coords0 = coords0[valid_m]
        coords1 = coords1[valid_m]
        select_conf = select_conf[valid_m]
        
        coords0 = coords0 * 2
        coords1 = coords1 * 2
        
        return coords0, coords1, reverse, select_conf
    