import os
import time
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from models.model import Matcher
from datasets.megadepth_scannet.data import build_distributed_dataset
from configs.data_config import *

cfg = {}
cfg["batch_size"] = 4
cfg['epochs'] = 30
cfg["end_lr"] = 1e-7
cfg["warming_up_epochs"] = 0
cfg['milestones'] = [3, 6, 9, 12, 17, 20, 23, 26, 29]
cfg["weight_decay"] = 1e-2
cfg["weights_dir"] = "weights/scannet"
cfg["restore_weight_path"] = ''
cfg['seed'] = 66

cfg['world_size'] = 2 # each process take one gpu, so world_size = num_nodes * num_gpus_per_node
cfg['node_rank'] = 0
cfg['master_addr'] = 'localhost'
cfg['master_port'] = 7777
cfg["init_lr"] =  6e-3 / 64 * cfg['batch_size'] * cfg['world_size']
megadepth_model_config = {
    'dim_conv_stem' : 64,
    "dims" : [128,192,256],
    "depths" : [0,1,2],
    "dropout" : 0.1,
    "d_spatial" : 32,
    "d_channel" : 128,
    "mbconv_expansion_rate":[1,1,1,2],
    "attn_depth": 9,
    "attn_name" : 'MultiScaleAttentionBlock',
    "attention" : 'focuse'
}
cfg['model'] = megadepth_model_config


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    
def save_weights(model,weight_dir,model_name,epoch,loss,optimizer,lr_scheduler):
    checkpoints = {
        'epoch':epoch,
        "model":model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler":lr_scheduler.state_dict()
    }
    weight_path = '{}/{}_{}_{:.4f}.pth'.format(weight_dir,model_name,epoch,loss)
    torch.save(checkpoints, weight_path)
    return weight_path
  
def tocuda(data):
    for k,v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda()
            
def train(local_rank,cfg):
    torch.cuda.set_device(local_rank)
    world_size = cfg['world_size']
    local_size = torch.cuda.device_count()
    rank = local_rank + cfg['node_rank'] * local_size
    random_seed(cfg['seed'] + rank * 100)
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://{}:{}".format(cfg['master_addr'], cfg['master_port']),
        rank=rank, 
        world_size=world_size)
    

    if rank == 0:
        if not os.path.exists(cfg["weights_dir"]):
            os.makedirs(cfg["weights_dir"])

    dataset_train,distributed_sampler = build_distributed_dataset(
        scannet_train_root_dir,
        scannet_train_val_npz_root,
        scannet_train_list,
        scannet_train_intrinsic_path,
        'train', 
        seed=cfg['seed'] + rank * 100,
        rank=rank, 
        world_size=world_size,
        score=0.4,
        n_samples_per_subset=200,
        augment_fn=None
    )
    
    train_loaders = DataLoader(
        dataset_train,
        batch_size=cfg['batch_size'],
        sampler=distributed_sampler,
        num_workers=1,
        pin_memory=True
    )
    warming_up_steps = len(train_loaders) * cfg['warming_up_epochs']
    
    steps_per_epoch = len(train_loaders)
    total_steps = steps_per_epoch * cfg['epochs']
    if rank == 0:
        print("num steps per epoch:", steps_per_epoch)
        print("total steps:", total_steps)

    # define model
    model = Matcher(cfg['model']).to(local_rank)
    torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank],output_device=local_rank)
    model.train()
    
    train_parameters = list(model.parameters())
    warm_up_optimizer = torch.optim.AdamW(train_parameters, lr=cfg["init_lr"],weight_decay=cfg["weight_decay"])
    warm_up_lr_scheduler = torch.optim.lr_scheduler.LinearLR(warm_up_optimizer,0.1,1,warming_up_steps)

    optimizer = torch.optim.AdamW(train_parameters,lr=cfg['init_lr'], weight_decay=cfg['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=0.5)


    trained_epochs = 0
    if os.path.exists(cfg["restore_weight_path"]):
        ckpts = torch.load(cfg["restore_weight_path"])
        model.load_state_dict(ckpts['model'])
        print('restore model!')
        trained_epochs = ckpts['epoch'] + 1
        if trained_epochs < cfg["warming_up_epochs"]:
            warm_up_optimizer.load_state_dict(ckpts["optimizer"])
            warm_up_lr_scheduler.load_state_dict(ckpts["lr_scheduler"])
            print("restore warming_up_optimizer and warmming_up_lr_scheduler!")
        else:
            optimizer.load_state_dict(ckpts["optimizer"])
            lr_scheduler.load_state_dict(ckpts["lr_scheduler"])
            print("restore optimizer and lr_scheduler")

    moving_mean_loss = 0
    momentum = 0.95
    
    optimizer.step()
    for _ in range(trained_epochs):
        lr_scheduler.step()
    
    iter_step = steps_per_epoch * trained_epochs
    for epoch in range(trained_epochs+1, cfg['epochs']+1):
        if epoch == 0:
            init_weight_path = os.path.join(cfg['weights_dir'],'init_weight.pth')
            if rank == 0:
                torch.save(model.state_dict(),init_weight_path)
            dist.barrier()
            model.load_state_dict(torch.load(init_weight_path,map_location='cuda:{}'.format(local_rank)))
            if rank == 0:
                print("sync the init weights")
                os.remove(init_weight_path)

        
        for i,data in enumerate(train_loaders):
            start = time.time()
            tocuda(data)
            loss_coarse, loss_fine = model.forward(data)
            loss = loss_coarse + loss_fine
            if epoch <= cfg["warming_up_epochs"]:
                using_optimizer = warm_up_optimizer
                using_scheduler = warm_up_lr_scheduler
            else:
                using_optimizer = optimizer
                using_scheduler = lr_scheduler


            using_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_parameters,max_norm=10,norm_type=2)
            using_optimizer.step()
            if epoch <= cfg["warming_up_epochs"]:
                using_scheduler.step()

            loss_val = loss.item()
            moving_mean_loss = moving_mean_loss * momentum + loss_val * (1-momentum)
            
            if rank == 0:
                if iter_step % 20 == 0:
                    print("step:{}, loss:{:.3f}, movin_mean_loss: {:.3f}, loss_coarse:{:.3f}, loss_fine:{:.3f}, lr: {:.10f}, cost_time:{:.3f}" \
                        .format(iter_step,loss_val, moving_mean_loss,loss_coarse.item(), loss_fine.item(),using_scheduler._last_lr[0],time.time() - start),flush=True)
                    
            iter_step += 1
            dist.barrier()

        if epoch >= cfg['warming_up_epochs']:
            lr_scheduler.step()
            
        if rank == 0:
            path = save_weights(model,cfg["weights_dir"],'model',epoch,moving_mean_loss,using_optimizer,using_scheduler)
            print("save_weights to %s"%path)
            
        torch.cuda.empty_cache()


if __name__ == "__main__":    
    mp.spawn(
        train,
        args=(cfg,),
        nprocs=cfg['world_size'],
        join=True
    )