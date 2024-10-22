import numpy as np
import cv2
import torch

from models.model import Matcher

cfg = {}
cfg["device"] = "cuda:0" if torch.cuda.is_available() else "cpu:0"
# cfg['device'] = 'cpu:0'
cfg['img_resize'] = 480
cfg['batch_size'] = 1
cfg['thresh'] = 0.2
cfg['high_thresh'] = 0.6
cfg['allow_reverse'] = False
cfg['add_matches'] = 0
cfg['iter_optimize'] = False
cfg['rplus'] = 1

cfg['weight_path'] = '/data2/ML/ImageMatcher/weights/20240326_megadepth_d7_refine_steps_v11_v2_832-9_multiscale/model_28_0.2728.pth'

model_config = {
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

def totensor(data,device):
    for k,v in data.items():
        if isinstance(v,torch.Tensor):
            data[k] = v.to(device)

def draw_image(img0,img1,qs,rs,mask=None,skip=50):
    h0,w0,c0 = img0.shape
    h1,w1,c1 = img1.shape
    assert c0 == c1,'assert error'
    oimg = np.zeros((max(h0,h1),w0+w1,c0),np.uint8)
    oimg[:h0,:w0,:] = img0
    oimg[:h1,w0:w0+w1,:] = img1
    green = (0,255,0)
    red = (0,0,255)
    for i,(q,r) in enumerate(zip(qs,rs)):
        if i % skip == 0:
            if mask is None:
                color = green 
            else:
                if mask[i]:
                    color = green
                else:
                    color = red
            cv2.line(oimg,(int(q[0]),int(q[1])),(int(r[0]+w0),int(r[1])),color,1)
    return oimg 

def read_image(path,img_resize=1216):
    img = cv2.imread(path,0)
    h,w = img.shape[:2]
    padded = np.zeros((img_resize,img_resize),dtype=np.uint8)
    scale = min(img_resize/h, img_resize/w)
    nh,nw = round(h*scale), round(w*scale)
    rimg = cv2.resize(img,(nw,nh))
    padded[:nh,:nw] = rimg
    return padded, 1/scale, h, w

@torch.no_grad()
def demo(cfg):
    model = Matcher(model_config).to(cfg['device'])
    model.eval()
    ckpts = torch.load(cfg['weight_path'],map_location=cfg['device'])['model']
    ckpts = {k.partition('module.')[2] : v for k,v in ckpts.items()} # comment out this line if the weight is not trained by ddp
    model.load_state_dict(ckpts)
    
    img0,scale0,h0,w0 = read_image('demo/1.jpg')
    img1,scale1,h1,w1 = read_image('demo/2.jpg')
    
    data = {
            'image0' : torch.from_numpy(img0).to(cfg['device'])[None,None] / 255,
            'image1' : torch.from_numpy(img1).to(cfg['device'])[None,None] / 255,
            }
    with torch.no_grad():
        src_pts,dst_pts,_,_ = model.forward_test(data,
                                                 thresh=cfg['thresh'],
                                                 high_thresh=cfg['high_thresh'],
                                                 allow_reverse=cfg['allow_reverse'],
                                                 add_matches=cfg['add_matches'],
                                                 iter_optimize=cfg['iter_optimize'],
                                                 rplus=1)
    
    src_pts = (src_pts).cpu().numpy()
    dst_pts = (dst_pts).cpu().numpy()
    
    # remove matches outside of the orig image
    c0 = src_pts * scale0
    c1 = dst_pts * scale1
    m0 = (c0 > 0).all(axis=-1) & (c0[:,0] < w0) & (c0[:,1]<h0)
    m1 = (c1 > 0).all(axis=-1) & (c1[:,0] < w1) & (c1[:,1]<h1)
    m = m0 & m1
    src_pts = src_pts[m]
    dst_pts = dst_pts[m]

    img0 = np.tile(img0[...,None],(1,1,3))
    img1 = np.tile(img1[...,None],(1,1,3))
    pred = draw_image(img0,img1, src_pts, dst_pts,None,skip=1)
    cv2.imwrite('demo/output.jpg',pred)
    
        
if __name__ == "__main__":
    demo(cfg)
