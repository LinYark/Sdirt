import torch
import numpy as np
from .metrics import *
import cv2 as cv
import torch
from torchvision.utils import save_image
import logging
import matplotlib.pyplot as plt

def select_focus_dist(depth, num, mode='linear', center=True, foc_d=1.0):
    """ Select focus distances from depth map. 

    Args:
        depth (tensor): [B, 1, H, W] tensor.
        num (int): focal stack size.
        mode (str, optional):
            "linear": sample linearly to approximate real-world applications.
            "importance": sample more data from close distances.
    """
    # assert num > 3, 'Focal stack size is too small'
    B, C, H, W = depth.shape
    mask = (depth > 0)
    
    focus_dists = (torch.ones((B,1),dtype=torch.float32).cuda())*foc_d # foc_d for 1.0m
    return focus_dists

    avg_depth = torch.sum(depth, dim=(1,2,3)) / torch.sum(mask, dim=(1,2,3))
    depth_max = torch.amax(depth, dim=(1,2,3))
    depth_min = torch.zeros_like(depth_max)
    for i in range(B):
        mask0 = mask[i,...]
        depth0 = depth[i, ...]
        depth_min[i] = torch.min(depth0[mask0>0])

    # Select focus distances
    if mode == 'linear':
        focus_dists = []
        for i in range(num):
            focus_dists.append(depth_min + i * (depth_max - depth_min) / (num - 1))

    elif mode == 'importance':
        focus_dists = [depth_max, depth_min]
        num = num - 2

        while len(focus_dists) < num:
            focus_dist = np.random.rand() * (depth_max - depth_min) + depth_min
            if focus_dist > avg_depth:
                accept_rate = (depth_max - focus_dist) / (depth_max - avg_depth)
            else:
                accept_rate = (focus_dist - depth_min) / (avg_depth - depth_min)

            accept = np.random.rand()
            if accept < accept_rate:
                focus_dists.append(focus_dist)
    else:
        raise NotImplementedError
    focus_dists = torch.stack(focus_dists, dim=1)
    focus_dists = torch.sort(focus_dists, dim=-1)[0]
    # focus_dists = (torch.ones((B,1)).cuda())
    return focus_dists

class ResultsMonitor():
    def __init__(self, train_mode):
        self.create_scores()
        self.train_mode = train_mode

    def create_scores(self):
        # Score for depth prediction
        self.Avg_abs_rel = 0.0
        self.Avg_sq_rel = 0.0
        self.Avg_mse = 0.0
        self.Avg_mae = 0.0
        self.Avg_rmse = 0.0
        self.Avg_rmse_log = 0.0
        self.Avg_accuracy_1 = 0.0
        self.Avg_accuracy_2 = 0.0
        self.Avg_accuracy_3 = 0.0
        self.Avg_accuracy_1_est, self.Avg_accuracy_2_est, self.Avg_accuracy_3_est = 0.0, 0.0, 0.0
        self.Avg_accuracy_1_fix, self.Avg_accuracy_2_fix, self.Avg_accuracy_3_fix = 0.0, 0.0, 0.0

        # Score for aif prediction
        self.Avg_psnr_deblur = 0.0
        self.Avg_ssim_deblur = 0.0

    def set_outputs(self, test_outputs):

        self.gt_aif = test_outputs['gt_aif']
        self.gt_depth = test_outputs['gt_depth']
        self.gt_depth = np.squeeze(self.gt_depth.data.cpu().numpy())
        self.test_mask = self.gt_depth > 1e-9

        self.gt_l = test_outputs['gt_l']
        self.gt_r = test_outputs['gt_r']

        self.rt_render_l = test_outputs['rt_render_l']
        self.rt_render_r = test_outputs['rt_render_r']

        self.pred_depth_est = test_outputs['pred_depth_est']
        self.pred_depth_est = np.squeeze(self.pred_depth_est.data.cpu().numpy())
        self.pred_depth_est[self.pred_depth_est<0] = 0 

        if self.train_mode=='deblur':
            self.pred_aif = test_outputs['pred_aif']
            self.pred_depth_fix = test_outputs['pred_depth_fix']
            self.pred_depth_fix = np.squeeze(self.pred_depth_fix.data.cpu().numpy())
            self.pred_depth_fix[self.pred_depth_fix<0] = 0   

    def compute_metrics(self, ):
        self.Avg_abs_rel = self.Avg_abs_rel + mask_abs_rel(self.pred_depth_est, self.gt_depth, self.test_mask)
        self.Avg_sq_rel = self.Avg_sq_rel + mask_sq_rel(self.pred_depth_est, self.gt_depth, self.test_mask)
        self.Avg_mse = self.Avg_mse + mask_mse(self.pred_depth_est, self.gt_depth, self.test_mask)
        self.Avg_mae = self.Avg_mae + mask_mae(self.pred_depth_est, self.gt_depth, self.test_mask)
        self.Avg_rmse = self.Avg_rmse + mask_rmse(self.pred_depth_est, self.gt_depth, self.test_mask)
        self.Avg_rmse_log = self.Avg_rmse_log + mask_rmse_log(self.pred_depth_est, self.gt_depth, self.test_mask)
        self.Avg_accuracy_1_est = self.Avg_accuracy_1_est + mask_accuracy_k(self.pred_depth_est, self.gt_depth, 1, self.test_mask)
        self.Avg_accuracy_2_est = self.Avg_accuracy_2_est + mask_accuracy_k(self.pred_depth_est, self.gt_depth, 2, self.test_mask)
        self.Avg_accuracy_3_est = self.Avg_accuracy_3_est + mask_accuracy_k(self.pred_depth_est, self.gt_depth, 3, self.test_mask)

        if self.train_mode=='deblur':
            self.Avg_accuracy_1_fix = self.Avg_accuracy_1_fix + mask_accuracy_k(self.pred_depth_fix, self.gt_depth, 1, self.test_mask)
            self.Avg_accuracy_2_fix = self.Avg_accuracy_2_fix + mask_accuracy_k(self.pred_depth_fix, self.gt_depth, 2, self.test_mask)
            self.Avg_accuracy_3_fix = self.Avg_accuracy_3_fix + mask_accuracy_k(self.pred_depth_fix, self.gt_depth, 3, self.test_mask)
            self.Avg_psnr_deblur = self.Avg_psnr_deblur + mask_psnr(self.pred_aif, self.gt_aif)
            self.Avg_ssim_deblur = self.Avg_ssim_deblur + mask_ssim(self.pred_aif, self.gt_aif)


    def save_images(self, result_img_dir, scene, idx, colorbar_flg=False):
        #save images
        if self.gt_aif is not None: save_image(self.gt_aif, f'{result_img_dir}/{scene}_{idx}_rgb_gt_aif.png', normalize=False)
        if self.gt_l is not None: save_image(self.gt_l, f'{result_img_dir}/{scene}_{idx}_rgb_gt_l.png', normalize=False)
        if self.gt_r is not None: save_image(self.gt_r, f'{result_img_dir}/{scene}_{idx}_rgb_gt_r.png', normalize=False)
        if self.rt_render_l is not None: save_image(self.rt_render_l, f'{result_img_dir}/{scene}_{idx}_rgb_rt_l.png', normalize=False)
        if self.rt_render_r is not None: save_image(self.rt_render_r, f'{result_img_dir}/{scene}_{idx}_rgb_rt_r.png', normalize=False)

        depth_max = self.gt_depth.max()*1.25
        gt_depth_save = (self.gt_depth / depth_max * 255.).astype(np.uint8)
        cv.imwrite(f'{result_img_dir}/{scene}_{idx}_depth_gt.png', cv.applyColorMap(gt_depth_save, cv.COLORMAP_JET))
        self.pred_depth_est = (self.pred_depth_est / depth_max * 255.).astype(np.uint8)
        cv.imwrite(f'{result_img_dir}/{scene}_{idx}_depth_est.png', cv.applyColorMap(self.pred_depth_est, cv.COLORMAP_JET))

        if self.train_mode=='deblur':
            save_image(self.pred_aif, f'{result_img_dir}/{scene}_{idx}_rgb_pred_aif.png', normalize=False)
            self.pred_depth_fix = (self.pred_depth_fix / depth_max * 255.).astype(np.uint8)
            cv.imwrite(f'{result_img_dir}/{scene}_{idx}_depth_fix.png', cv.applyColorMap(self.pred_depth_fix, cv.COLORMAP_JET))

        file_info = [result_img_dir, scene, idx]
        # self.save_infocus_with_red_color(file_info)
        if colorbar_flg:
            self.colorbar(file_info)

    def save_infocus_with_red_color(self,file_info):
        result_img_dir, scene, idx = file_info
        draw_focus_line = False
        if draw_focus_line:
            gt_depth_t = torch.from_numpy(self.gt_depth).to(dtype=self.rt_render_l.dtype, device=self.rt_render_l.device)
            mask = (gt_depth_t >= 0.98) & (gt_depth_t <= 1.02)    # shape: (H, W)
            mask = mask.unsqueeze(0).unsqueeze(0)                 # shape: (1, 1, H, W)
            mask = mask.expand(-1, 3, -1, -1)                      # shape: (1, 3, H, W)
            red = torch.tensor([1.0, 0.0, 0.0], device=self.rt_render_l.device).view(1, 3, 1, 1)
            alpha = 0.2  
            if self.rt_render_l is not None:
                blended = (1 - alpha) * self.rt_render_l + alpha * red
                rt_render_l = torch.where(mask, blended, self.rt_render_l)
                save_image(rt_render_l, f'{result_img_dir}/{scene}_{idx}_rgb_focus_l.png', normalize=False)
            if rt_render_r is not None: 
                blended = (1 - alpha) * self.rt_render_r + alpha * red
                rt_render_r = torch.where(mask, blended, self.rt_render_r)
                save_image(rt_render_r, f'{result_img_dir}/{scene}_{idx}_rgb_focus_r.png', normalize=False)

    def colorbar(self, file_info):
        result_img_dir, scene, idx = file_info
        gt_depth = self.gt_depth*1.0
        pred_depth_est = self.pred_depth_est*1.0

        show_mode='simple' # 'draw'
        if show_mode ==0:
            plt.imshow(gt_depth, cmap='jet',vmin=0)
            cbar = plt.colorbar()
            cbar.set_label("Meter") 
            plt.savefig(f'{result_img_dir}/{scene}_{idx}_plt_gt.png', bbox_inches='tight', dpi=300)
            plt.close()
            plt.imshow(pred_depth_est, cmap='jet',vmin=0)
            cbar = plt.colorbar()
            cbar.set_label("Meter") 
            plt.savefig(f'{result_img_dir}/{scene}_{idx}_plt_pred.png', bbox_inches='tight', dpi=300)
            plt.close()

        if show_mode == 'draw':
            gt_depth[gt_depth < 0.3] = 0
            min_v = gt_depth[gt_depth>0.3].min()
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(gt_depth, cmap="jet",vmin=min_v)#,vmax=depth_max,vmin=0
            ax.set_xticks([])
            ax.set_yticks([])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.06)
            cbar = fig.colorbar(im, cax=cax)#,format='%.1f'
            # cbar.set_label('Meter', fontsize=16)
            cbar.ax.tick_params(labelsize=16)
            plt.savefig(f'{result_img_dir}/{scene}_{idx}_plt_gt.png', bbox_inches='tight', dpi=300)
            plt.close()
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(pred_depth_est, cmap="jet")#,vmax=depth_max,vmin=0
            ax.set_xticks([])
            ax.set_yticks([])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.06)
            cbar = fig.colorbar(im, cax=cax)#,format='%.1f'
            # cbar.set_label('Meter', fontsize=16)
            cbar.ax.tick_params(labelsize=16)
            plt.savefig(f'{result_img_dir}/{scene}_{idx}_plt_pred.png', bbox_inches='tight', dpi=300)
            plt.close()

    def logging(self, epoch, num_scene):
        # logging.info(f"Avg_abs_sq_rmse_log({epoch}): {self.Avg_abs_rel / num_val}, {self.Avg_sq_rel / num_val}, {self.Avg_rmse / num_val}, {self.Avg_rmse_log / num_val}")
        logging.info(f"Avg_mse/mae({epoch}): {self.Avg_mse / num_scene}, {self.Avg_mae / num_scene}")
        logging.info(f"Avg_acc_est({epoch}): {self.Avg_accuracy_1_est / num_scene}, {self.Avg_accuracy_2_est / num_scene}, {self.Avg_accuracy_3_est / num_scene}")
        if self.train_mode == 'deblur':
            logging.info(f"Avg_ps_deblur({epoch}): {self.Avg_psnr_deblur / num_scene} {self.Avg_ssim_deblur / num_scene}")
            logging.info(f"Avg_acc_fix({epoch}): {self.Avg_accuracy_1_fix / num_scene}, {self.Avg_accuracy_2_fix / num_scene}, {self.Avg_accuracy_3_fix / num_scene}")

    def save_pth(self, args, scene, num_scene, net):
        if f'mse_{scene}_min' not in args:
            args[f'mse_{scene}_min'] = 100.0
        if f'acc1_{scene}_max' not in args:
            args[f'acc1_{scene}_max'] = 0

        torch.save(net.state_dict(), f'{args["results_dir"]}/depth_net_last.pkl')
        # if self.Avg_mse / num_scene < args[f'mse_{scene}_min']:
        #     args[f'mse_{scene}_min'] = self.Avg_mse / num_scene
        #     torch.save(net.state_dict(), f'{args["results_dir"]}/{scene}_net_best_mse.pkl')
        if self.Avg_accuracy_1_est / num_scene > args[f'acc1_{scene}_max']:
            args[f'acc1_{scene}_max'] = self.Avg_accuracy_1_est / num_scene
            torch.save(net.state_dict(), f'{args["results_dir"]}/{scene}_net_best_acc1.pkl')