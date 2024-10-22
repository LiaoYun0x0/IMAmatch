import os
from torch import distributed as dist
from torch.utils.data import ConcatDataset

from .megadepth import MegaDepthDataset
from .scannet import ScanNetDataset
from .sampler import RandomConcatSampler,DitributedRandomConcatSampler

def build_dataset(root_dir,list_root,list_path,intrinsic_path,mode,img_resize,score=0.4,seed=66,max_data_size=1e8,n_samples_per_subset=100,augment_fn=None):
    with open(list_path,'r') as txt:
        npz_names = txt.readlines()
    datasets = []
    count_size = 0
    for npz_name in npz_names:
        if count_size > max_data_size:
            break
        npz_name = npz_name.strip('\n')
        if not npz_name.endswith('.npz'):
            npz_name += '.npz'
        npz_path = os.path.join(list_root,npz_name)
        if 'MegaDepth' in root_dir:
            dataset = MegaDepthDataset(root_dir, npz_path, mode, 
                                            min_overlap_score=score,
                                            img_resize=img_resize, 
                                            img_padding=True,
                                            depth_padding=True,
                                            df=8,
                                            augment_fn=augment_fn)
        elif 'scannet' in root_dir:
            dataset = ScanNetDataset(root_dir,
                                   npz_path,
                                   intrinsic_path,
                                   mode=mode,
                                   min_overlap_score=score,
                                   augment_fn=None,
                                   pose_dir=None)
        else:
            raise NotImplementedError
        
        datasets.append(dataset)
        curr_size = len(datasets[-1])
        if count_size + curr_size <= max_data_size:
            count_size += curr_size
        else:
            load_size = int(max_data_size - count_size)
            count_size += load_size
            if 'MegaDepth' in root_dir:
                datasets[-1].pair_infos = datasets[-1].pair_infos[:load_size]
            elif 'scannet' in root_dir:
                datasets[-1].data_names = datasets[-1].data_names[:load_size]
            else:
                raise NotImplementedError()
            break
    dataset = ConcatDataset(datasets)
    sampler = RandomConcatSampler(dataset,n_samples_per_subset,True,True,1,seed)
    return dataset,sampler

def build_distributed_dataset(root_dir,list_root,list_path,intrinsic_path,mode,img_resize,seed,rank,world_size,score=0.4,max_data_size=1e8,n_samples_per_subset=100,augment_fn=None):
    with open(list_path,'r') as txt:
        npz_names = txt.readlines()
    
    remainder = len(npz_names) % world_size
    if remainder != 0:
        npz_names = npz_names + npz_names[:world_size-remainder]
        assert len(npz_names) % world_size == 0, 'num of scenes is not divible by world_size'
    npz_names = npz_names[rank:len(npz_names):world_size]
    datasets = []
    count_size = 0
    for npz_name in npz_names:
        # if count_size > max_data_size:
        #     break
        npz_name = npz_name.strip('\n')
        if not npz_name.endswith('.npz'):
            npz_name += '.npz'
        npz_path = os.path.join(list_root,npz_name)
        if 'MegaDepth' in root_dir:
            dataset = MegaDepthDataset(root_dir, npz_path, mode, 
                                            min_overlap_score=score,
                                            img_resize=img_resize, 
                                            img_padding=True,
                                            depth_padding=True,
                                            df=8,
                                            augment_fn=augment_fn)
        elif 'scannet' in root_dir:
            dataset = ScanNetDataset(root_dir,
                                   npz_path,
                                   intrinsic_path,
                                   mode=mode,
                                   min_overlap_score=score,
                                   augment_fn=None,
                                   pose_dir=None)
        else:
            raise NotImplementedError
        
        datasets.append(dataset)
        # curr_size = len(datasets[-1])
        # if count_size + curr_size <= max_data_size:
        #     count_size += curr_size
        # else:
        #     load_size = int(max_data_size - count_size)
        #     count_size += load_size
        #     if 'MegaDepth' in root_dir:
        #         datasets[-1].pair_infos = datasets[-1].pair_infos[:load_size]
        #     elif 'scannet' in root_dir:
        #         datasets[-1].data_names = datasets[-1].data_names[:load_size]
        #     else:
        #         raise NotImplementedError()
        #     break
    dataset = ConcatDataset(datasets)
    sampler = DitributedRandomConcatSampler(dataset,n_samples_per_subset,True,False,1,seed)
    return dataset,sampler

def build_test_dataset(root_dir,list_root,list_path,intrinsic_path,mode,img_resize,score=0.4,max_data_size=1e8):
    with open(list_path,'r') as txt:
        npz_names = txt.readlines()
    datasets = []
    count_size = 0
    for npz_name in npz_names:
        if count_size > max_data_size:
            break
        npz_name = npz_name.strip('\n')
        if not npz_name.endswith('.npz'):
            npz_name += '.npz'
        npz_path = os.path.join(list_root,npz_name)
        if 'MegaDepth' in root_dir:
            dataset = MegaDepthDataset(root_dir, npz_path, mode, 
                                            min_overlap_score=score,
                                            img_resize=img_resize, img_padding=True,depth_padding=True,df=8)
        elif 'scannet' in root_dir:
            dataset = ScanNetDataset(root_dir,
                                   npz_path,
                                   intrinsic_path,
                                   mode=mode,
                                   min_overlap_score=score,
                                   augment_fn=None,
                                   pose_dir=None)
        
        datasets.append(dataset)
        curr_size = len(datasets[-1])
        if count_size + curr_size <= max_data_size:
            count_size += curr_size
        else:
            load_size = int(max_data_size - count_size)
            count_size += load_size
            if 'MegaDepth' in root_dir:
                datasets[-1].pair_infos = datasets[-1].pair_infos[:load_size]
            elif 'scannet' in root_dir:
                datasets[-1].data_names = datasets[-1].data_names[:load_size]
            else:
                raise NotImplementedError()
    dataset = ConcatDataset(datasets)
    return dataset