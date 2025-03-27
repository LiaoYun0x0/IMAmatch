import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import Matcher
from datasets.megadepth_scannet.data import build_test_dataset
from configs.data_config import *

cfg = {}
cfg["device"] = "cuda:0" if torch.cuda.is_available() else "cpu:0"
cfg['img_resize'] = 1216
cfg['batch_size'] = 1
cfg['log_file'] = 'log/eval.log'
cfg['thresh'] = 0.2
cfg['high_thresh'] = 0.6
cfg['allow_reverse'] = True
cfg['add_matches'] = 4
cfg['iter_optimize'] = True
cfg['rplus'] = 1
cfg['use_ba'] = False       # If use the weight of bundle adjustment, the method will be slow.

cfg['weight_path'] = 'path-to-weight'


if cfg['use_ba']:
    try: 
        from ba.bundle_adjust_gauss_newton_2_view import normalize, run_bundle_adjust_2_view
    except:
        print('failed to use ba')

megadepth_model_config = {
    'dim_conv_stem' : 64,
    "dims" : [128,192,256],
    "depths" : [0,1,2],
    "dropout" : 0.1,
    "d_spatial" : 32,
    "d_channel" : 128,
    "mbconv_expansion_rate":[1,1,1,2],
    "attn_depth": 9,
    "attn_name" : "MultiScaleAttentionBlock",
    'img_size' : cfg['img_resize'],
    'attention':'focuse'
}

cfg['model'] = megadepth_model_config

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err

def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K0[1, 1], K1[0, 0], K1[1, 1]])
    E, mask = cv2.findEssentialMat(kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret

def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)
    return aucs

def totensor(data,device):
    for k,v in data.items():
        if isinstance(v,torch.Tensor):
            data[k] = v.to(device)
        
@torch.no_grad()
def eval(cfg):
    model = Matcher(cfg['model']).to(cfg['device'])
    model.eval()
    
    ckpts = torch.load(cfg['weight_path'],map_location=cfg['device'])['model']
    ckpts = {k.partition('module.')[2] : v for k,v in ckpts.items()}
    model.load_state_dict(ckpts)
    
    dataset_test = build_test_dataset(
        megadepth_root_dir, 
        megadepth_test_npz_root, 
        megadepth_test_list, 
        None, 
        'val', 
        cfg['img_resize'],
        score=0.0)
    eval_loaders =  DataLoader(
        dataset_test,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=4
    )
    print(len(eval_loaders))

    R_errors, t_errors,inliers = [],[],[]
    tbar = tqdm(eval_loaders)
    for i,data in enumerate(tbar):
        totensor(data, cfg['device'])
        with torch.no_grad():
            src_pts,dst_pts,reverse,mconf = model.forward_test(data,cfg['thresh'],
                                                         high_thresh=cfg['high_thresh'],
                                                         allow_reverse=cfg['allow_reverse'],
                                                         add_matches=cfg['add_matches'],
                                                         iter_optimize=cfg['iter_optimize'],
                                                         rplus=1)
        mconf = mconf.cpu().numpy()
        K0 = data['K0'][0].cpu().numpy()
        K1 = data['K1'][0].cpu().numpy()
        if not reverse:
            src_pts = (src_pts.cpu() * data['scale0'][0].cpu()).numpy()
            dst_pts = (dst_pts.cpu() * data['scale1'][0].cpu()).numpy()
            ret = estimate_pose(src_pts, dst_pts, K0, K1, thresh=0.5)
            if cfg['use_ba']: 
                # bundle to refine pose
                if ret is not None:
                    try:
                        inlier_mask = ret[2]
                        pred_T021 = torch.eye(4).unsqueeze(0).to(cfg['device'])
                        pred_T021[0, :3, 3] = torch.from_numpy(ret[1]).to(cfg['device']).unsqueeze(0)
                        pred_T021[0, :3, :3] = torch.from_numpy(ret[0]).to(cfg['device']).unsqueeze(0)
                        confidence = torch.from_numpy(mconf[inlier_mask]).to(cfg['device']).unsqueeze(0)
                        intr0 = torch.from_numpy(K0).to(cfg['device']).unsqueeze(0)
                        intr1 = torch.from_numpy(K1).to(cfg['device']).unsqueeze(0)
                        kpts0_norm = normalize(torch.from_numpy(src_pts[inlier_mask]).to(cfg['device']).unsqueeze(0), intr0)
                        kpts1_norm = normalize(torch.from_numpy(dst_pts[inlier_mask]).to(cfg['device']).unsqueeze(0), intr1)
                        pred_T021_refine, valid_refine = run_bundle_adjust_2_view(kpts0_norm, kpts1_norm, confidence.unsqueeze(-1), pred_T021, \
                                n_iterations=10)
                        pred_T021[valid_refine] = pred_T021_refine
                        ret = (pred_T021[0, :3, :3].cpu().numpy(), pred_T021[0, :3, 3].cpu().numpy(), inlier_mask) if pred_T021 is not None else None
                    except Exception as e:
                        print(e)
            if ret is None:
                R_errors.append(np.inf)
                t_errors.append(np.inf)
            else:
                R, t, _inliers = ret
                t_err, R_err = relative_pose_error(data['T_0to1'][0].cpu().numpy(), R, t, ignore_gt_t_thr=0.0)
                R_errors.append(R_err)
                t_errors.append(t_err)
                
        else:
            src_pts = (src_pts.cpu() * data['scale1'][0].cpu()).numpy()
            dst_pts = (dst_pts.cpu() * data['scale0'][0].cpu()).numpy()
            ret = estimate_pose(src_pts, dst_pts, K1, K0,thresh=0.5)
            if cfg['use_ba']:
                # bundle to refine pose
                if ret is not None:
                    try:
                        inlier_mask = ret[2]
                        pred_T021 = torch.eye(4).unsqueeze(0).to(cfg['device'])
                        pred_T021[0, :3, 3] = torch.from_numpy(ret[1]).to(cfg['device']).unsqueeze(0)
                        pred_T021[0, :3, :3] = torch.from_numpy(ret[0]).to(cfg['device']).unsqueeze(0)
                        confidence = torch.from_numpy(mconf[inlier_mask]).to(cfg['device']).unsqueeze(0)
                        intr0 = torch.from_numpy(K1).to(cfg['device']).unsqueeze(0)
                        intr1 = torch.from_numpy(K0).to(cfg['device']).unsqueeze(0)
                        kpts0_norm = normalize(torch.from_numpy(src_pts[inlier_mask]).to(cfg['device']).unsqueeze(0), intr0)
                        kpts1_norm = normalize(torch.from_numpy(dst_pts[inlier_mask]).to(cfg['device']).unsqueeze(0), intr1)
                        pred_T021_refine, valid_refine = run_bundle_adjust_2_view(kpts0_norm, kpts1_norm, confidence.unsqueeze(-1), pred_T021, \
                                n_iterations=10)
                        pred_T021[valid_refine] = pred_T021_refine
                        ret = (pred_T021[0, :3, :3].cpu().numpy(), pred_T021[0, :3, 3].cpu().numpy(), inlier_mask) if pred_T021 is not None else None
                    except Exception as e:
                        print(e)
                
                
            if ret is None:
                R_errors.append(np.inf)
                t_errors.append(np.inf)
            else:
                R, t, _inliers = ret
                t_err, R_err = relative_pose_error(data['T_1to0'][0].cpu().numpy(), R, t, ignore_gt_t_thr=0.0)
                R_errors.append(R_err)
                t_errors.append(t_err)
        
    
    R_errors = np.array(R_errors,dtype=np.float32)
    t_errors = np.array(t_errors,dtype=np.float32)
    thresholds = [5, 10, 20, 30, 40, 50, 60, 70]
    pose_errors = np.maximum(R_errors,t_errors)
    aucs = error_auc(pose_errors, thresholds)
    print(aucs)
        
if __name__ == "__main__":
    eval(cfg)
