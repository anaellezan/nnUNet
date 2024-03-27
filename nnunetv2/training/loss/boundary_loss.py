from typing import Callable

import torch
from torch import nn
from torch import einsum

import numpy as np
from scipy.ndimage import distance_transform_edt as eucl_distance
from skimage import segmentation as skimage_seg

class BoundaryLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, resolution: list = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1., ddp: bool = True):

        super(BoundaryLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.resolution = resolution
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]

        # make everything shape (b, c)                                                                                                                                      
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding                                                                                        
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)
        
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
                
        dist_map = torch.zeros_like(x)

        for i in range(len(y_onehot)):
            dist_map[i] = torch.from_numpy(self.compute_distance_map(y_onehot[i], self.resolution))

#        print(f"distance map mean = {dist_map.mean()}")
#        print(x)
        if len(shp_x) == 4:
            multipled = einsum("bkwh,bkwh->bkwh", x, dist_map)
        else:
            multipled = einsum("bkxyz,bkxyz->bkxyz", x, dist_map)
            
        loss = multipled.mean()

        return loss

    @staticmethod
    def compute_distance_map(seg, resolution):
        K: int = len(seg)
        seg = seg.cpu().numpy()
        res = np.zeros_like(seg, dtype=float)
        for k in range(K):
            posmask = seg[k].astype(np.bool)

            if posmask.any():
                negmask = ~posmask
                res[k] = eucl_distance(negmask, sampling=resolution)* negmask - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
  
        return res


class BoundaryLoss2(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, resolution: list = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1., ddp: bool = True):

        super(BoundaryLoss2, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.resolution = resolution
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]

        # make everything shape (b, c)                                                                                                                                      
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding                                                                                        
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)
        
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
                
        dist_map = torch.zeros_like(x)

        for i in range(len(y_onehot)):
            dist_map[i] = torch.from_numpy(self.compute_distance_map(y_onehot[i], self.resolution))

#        print(f"distance map mean = {dist_map.mean()}")
#        print(x)
        if len(shp_x) == 4:
            multipled = einsum("bkwh,bkwh->bkwh", x, dist_map)
        else:
            multipled = einsum("bkxyz,bkxyz->bkxyz", x, dist_map)
            
        loss = multipled.mean()

        return loss

    @staticmethod
    def compute_distance_map(seg, resolution):
        K: int = len(seg)
        seg = seg.cpu().numpy()
        res = np.zeros_like(seg, dtype=float)
        for k in range(K):
            posmask = seg[k].astype(np.bool)

            if posmask.any():
                res[k] = eucl_distance(posmask, sampling=resolution)
  
        return res


class BoundaryLossSDM(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, resolution: list = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1., ddp: bool = True):

        super(BoundaryLossSDM, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.resolution = resolution
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]

        # make everything shape (b, c)                                                                                                                                      
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding                                                                                        
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)
        
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
        
        dist_map= torch.from_numpy(self.compute_sdm(y_onehot, shp_x))

        if dist_map.device != x.device:
            dist_map = dist_map.to(x.device).type(torch.float32)
            
        if len(shp_x) == 4:
            multipled = einsum("bkwh,bkwh->bkwh", x, dist_map)
        else:
            multipled = einsum("bkxyz,bkxyz->bkxyz", x, dist_map)
            
        loss = multipled.mean()

        return loss
    
    @staticmethod
    def compute_sdm(img_gt, out_shape):
        """
        compute the signed distance map of binary mask
        input: segmentation, shape = (batch_size, x, y, z)
        output: the Signed Distance Map (SDM) 
        sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
        """
        img_gt = img_gt.cpu().numpy()
        img_gt = img_gt.astype(np.uint8)

        gt_sdf = np.zeros(out_shape)
        for b in range(out_shape[0]): # batch size
            for c in range(1, out_shape[1]): # channel
                posmask = img_gt[b][c].astype(np.bool)
                if posmask.any():
                    negmask = ~posmask
                    posdis = eucl_distance(posmask)
                    negdis = eucl_distance(negmask)
                    boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                    sdf = negdis - posdis
                    sdf[boundary==1] = 0
                    gt_sdf[b][c] = sdf

        return gt_sdf


class BoundaryLossRRW(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, resolution: list = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1., ddp: bool = True):

        super(BoundaryLossRRW, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.resolution = resolution
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]

        # make everything shape (b, c)                                                                                                                                      
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding                                                                                        
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)
        
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
        
        dist_map= torch.from_numpy(self.compute_rrw(y_onehot))

        if dist_map.device != x.device:
            dist_map = dist_map.to(x.device).type(torch.float32)
            
        if len(shp_x) == 4:
            multipled = einsum("bkwh,bkwh->bkwh", x, dist_map)
        else:
            multipled = einsum("bkxyz,bkxyz->bkxyz", x, dist_map)
            
        loss = multipled.mean()

        return loss
    
    @staticmethod
    def compute_rrw(img_gt):
        
       img_gt = img_gt.detach().cpu().numpy() 
       rrwmap = np.zeros_like(img_gt)
       for b in range(rrwmap.shape[0]):
           for c in range(rrwmap.shape[1]):
               rrwmap[b][c] = eucl_distance(img_gt[b][c])
               rrwmap[b][c] = -1 * (rrwmap[b][c] / (np.max(rrwmap[b][c] + 1e-15)))
       rrwmap[rrwmap==0] = 1
       
       return rrwmap
