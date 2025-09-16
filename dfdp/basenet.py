""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from .dddnet.dddnet import YRStereonet_3D, Mydeblur


class Basenet(nn.Module):
    def __init__(self, train_mode='dfdp'):
        super(Basenet, self).__init__()
        # train_mode = 'dfdp' or 'deblur'
        self.train_mode=train_mode
        
        self.dfdp_net = YRStereonet_3D()
        self.deblur_net = Mydeblur()
        
    @autocast()
    def forward(self, input_dict):
        losses, outputs = self.dfdp(input_dict, train=True)
        return losses, outputs
    
    def dfdp(self, input_dict, train=False):
        stack_rgb, gt_aif = input_dict['stack_rgb_img'], input_dict['AiF_img']
        rt_render_l, rt_render_r = stack_rgb[:,0:3,:,:], stack_rgb[:,3:,:,:]
        gt_depth = self.linear(input_dict['gt_depth'])

        depth_est = self.dfdp_net(rt_render_l, rt_render_r)
        if self.train_mode == 'deblur':
            depth_fix, aif_fix = self.deblur_net(rt_render_l, rt_render_r, depth_est)

        losses = None
        if train is True:
            results, gts = dict(), dict()
            gts['gt_depth'], gts['gt_aif'] = gt_depth, gt_aif
            gts['rt_render_l'], gts['rt_render_r'] = rt_render_l, rt_render_r
            results['pred_depth_est'] = depth_est
            if self.train_mode == 'deblur':
                results['pred_depth_fix'], results['pred_aif'] = depth_fix, aif_fix
            losses = self.compute_loss(results, gts)

        outputs=dict()
        outputs['gt_depth'], outputs['gt_aif'] = self.inverse_linear(gt_depth), gt_aif
        outputs['gt_l'], outputs['gt_r'] = None, None
        outputs['rt_render_l'], outputs['rt_render_r'] = rt_render_l, rt_render_r
        outputs['pred_depth_est'] = self.inverse_linear(depth_est.to(torch.float32))
        if self.train_mode == 'deblur':
            outputs['pred_depth_fix'], outputs['pred_aif'] = self.inverse_linear(depth_fix.to(torch.float32)), aif_fix
        return losses, outputs

    def compute_loss(self, results, gts):
        losses = dict()
        l2 = nn.MSELoss(reduction='mean')
        l1 = nn.SmoothL1Loss(reduction='mean')

        gt_depth, gt_aif = gts['gt_depth'], gts['gt_aif']
        rt_render_l, rt_render_r = gts['rt_render_l'], gts['rt_render_r']
        pred_depth_est = results['pred_depth_est']

        losses['depth_est'] = l1(pred_depth_est[self.mask], gt_depth[self.mask])
        losses['total'] = losses['depth_est']

        if self.train_mode == 'deblur':
            pred_depth_fix, pred_aif  = results['pred_depth_fix'], results['pred_aif']
            losses['depth_fix'] = l1(pred_depth_fix[self.mask], gt_depth[self.mask])
            losses['aif'] = l1(pred_aif, gt_aif)
            losses['total'] = losses['depth_est']*2 + losses['depth_fix'] + losses['aif']
        return losses 
    
    def inference(self, input_dict):
        gt_depth, gt_aif = self.linear(input_dict['depth']), input_dict['AiF_img']
        stack_rgb = input_dict['stack_rgb_img']
        gt_l, gt_r = stack_rgb[:,0:3,:,:], stack_rgb[:,3:,:,:]

        depth_est = self.dfdp_net(gt_l, gt_r)
        if self.train_mode == 'deblur':
            depth_fix, aif_fix = self.deblur_net(gt_l, gt_r, depth_est)

        outputs=dict()
        outputs['gt_depth'], outputs['gt_aif'] = self.inverse_linear(gt_depth), gt_aif
        outputs['gt_l'], outputs['gt_r'] = gt_l, gt_r
        outputs['rt_render_l'], outputs['rt_render_r'] = None, None
        outputs['pred_depth_est'] = self.inverse_linear(depth_est.to(torch.float32), mask=False)
        if self.train_mode == 'deblur':
            outputs['pred_depth_fix'], outputs['pred_aif'] = self.inverse_linear(depth_fix.to(torch.float32), mask=False), aif_fix
        return outputs

    def robust_l1(self, x):
        """Robust L1 metric."""
        return (x**2 + 0.001**2)**0.5
    
    def linear(self, depth):
        self.mask = depth > 1e-9
        self.mask.detach_()
        depth[self.mask] = torch.log(depth[self.mask])
        return depth

    def inverse_linear(self, depth, mask=None):
        if mask is None:
            depth[self.mask] = torch.exp(depth[self.mask])
        else:
            depth = torch.exp(depth)
        return depth


    