""" Simulating Dual-Pixel Images From Ray Tracing For Depth Estimation.
    
Real lens for training, real lens for testing: our Sdirt can generalize well in the real world with only synthetic data.

Use DfDP Network for training and evaluation.
"""
import os
import sys
sys.dont_write_bytecode = True
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
ddp = False ##CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=gpu  --master_port=12357 3_cycle_net.py 
if ddp == False:
    os.environ["CUDA_VISIBLE_DEVICES"]='1'

import yaml
import time
import logging
import cv2 as cv
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

from deeplens.utils import set_seed, set_logger
from deeplens.psfnet import *
from dfdp import *

def ddp_setup():
    os.environ["MASTER_ADDR"] = "localhost"
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def config():
    with open('configs/dfdp_by_sdirt_rf50mm.yml') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    
    # Device
    num_gpus = torch.cuda.device_count()
    args['num_gpus'] = num_gpus
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    args['device'] = device
    logging.info(f'Using {num_gpus} GPUs')

    # Result folder
    result_dir = f'./results/' + datetime.now().strftime("%m%d-%H%M%S") + '-Sdirt_dev'
    args['results_dir'] = result_dir
    os.makedirs(result_dir, exist_ok=True)
    logging.info(f'Result folder: {result_dir}')
    
    args['train_mode'] = 'dfdp' # 'cycle' 'render'
    
    # Random seed
    set_seed(123456)
    torch.set_default_dtype(torch.float32)
    
    return args

def train(args):
    if ddp : ddp_setup()
    id = 0 if ddp is False else int(os.environ['LOCAL_RANK'])
    if id == 0:  # Logger
        set_logger(args['results_dir'])

    device = args['device']

    # Lens
    train_lens, test_lens = get_lens(args)
    
    dfdp_net = Basenet(args['train_mode'])
    dfdp_net = dfdp_net.to(device)
    if ddp:
        dfdp_net = DDP(dfdp_net, device_ids=[id], find_unused_parameters=True)
    else:
        dfdp_net = nn.DataParallel(dfdp_net)
    
    if 'dfdpnet_pretrained' in args['train'].keys() :
        net_dict = dfdp_net.state_dict()
        pretrain_dict = torch.load(args['train']['dfdpnet_pretrained'])
        update_dict = {}
        for k,v in pretrain_dict.items():
            k1 = k #.replace('net.','net.backbone.')
            if k1 in net_dict and net_dict[k1].shape == pretrain_dict[k].shape:
                update_dict[k1]=v
        net_dict.update(update_dict)
        dfdp_net.load_state_dict(net_dict)
    torch.cuda.empty_cache()
    dfdp_net = dfdp_net.to(device)

    # Only test on sample real_set of DP119
    flat_sample = get_flat_sample_set(args)
    box_sample, flat2depth_sample, casual_sample = get_depth_sample_set(args)
    test_DP_images(test_lens, flat_sample, 'flatSample', args)
    test(dfdp_net, box_sample, "boxSample", args)
    test(dfdp_net, flat2depth_sample, "f2dSample", args)
    test(dfdp_net, casual_sample, "casualSample", args)
    exit()


    # Dataset
    nyu_fs_train_set, nyu_train_set, val_set = get_dataset(args)
    print(f'Totally {len(nyu_fs_train_set)} images for training, {len(val_set)} images for test.')
    if ddp==False:
        nyu_fs_train_loader = DataLoader(nyu_fs_train_set, batch_size=args['bs'], num_workers=4, pin_memory=True, shuffle=True,drop_last=True)
        nyu_train_loader = DataLoader(nyu_train_set, batch_size=args['bs'], num_workers=4, pin_memory=True, shuffle=True,drop_last=True)
    else:
        nyu_fs_train_loader = DataLoader(dataset=nyu_fs_train_set, batch_size=args['bs'], shuffle=False, pin_memory=True, drop_last=True, sampler=DistributedSampler(nyu_fs_train_set, drop_last=True))
        nyu_train_loader = DataLoader(dataset=nyu_train_set, batch_size=args['bs'], shuffle=False, pin_memory=True, drop_last=True, sampler=DistributedSampler(nyu_train_set, drop_last=True))

    optimizer = optim.AdamW(dfdp_net.parameters(), lr=float(args['lr']))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs']*len(nyu_fs_train_set), eta_min=0)
    scaler = GradScaler()

    flat_set = get_flat_test_set(args)
    box_set, flat2depth_set, casual_set = get_depth_test_set(args)

    # Only test on full real_set DP119
    test_DP_images(test_lens, flat_set, 'flat', args)
    test(dfdp_net, box_set, "box", args)
    test(dfdp_net, flat2depth_set, "f2d", args)
    test(dfdp_net, casual_set, "casual", args)
    exit()


    # Training
    for epoch in range(args['epochs'] + 1):
        if ddp: 
            nyu_fs_train_loader.sampler.set_epoch(epoch)
            nyu_train_loader.sampler.set_epoch(epoch)

        # Evaluation
        if id == 0  : # and epoch > 0 
            with torch.no_grad():
                validate(dfdp_net, test_lens, val_set, 'fs', args, epoch)
                # test(dfdp_net, flat2depth_set, "flat", args, epoch)
                test(dfdp_net, box_set, "box", args, epoch)
                # test(dfdp_net, casual_set, "casual", args, epoch)
                logging.info("\n")

        # train set use nyu+fs, and finetune in only nyu set (cause fs is a virtual RGBD set)
        train_loader = nyu_fs_train_loader if epoch <= args['epochs']//2 else nyu_train_loader

        # Training
        dfdp_net.train()
        for sample in tqdm(train_loader, dynamic_ncols=True):
            # Input data
            aif, gt_depth = sample
            aif = aif.to(device)
            gt_depth = gt_depth.to(device)    # real depth in [m]

            input_dict = {'gt_depth':gt_depth, 'AiF_img':aif}
            # Render focal stack
            with torch.no_grad():
                with autocast():
                    # Select random focus distance
                    focus_dists = select_focus_dist(gt_depth, args['n_stack'], mode='linear') # n=1 in dfdp experiment
                    # Simulate focal stack
                    focal_stack = []
                    for i in range(args['bs']):
                        foc_dist = focus_dists[i:i+1, 0]
                        defocus_img = train_lens.render(aif[i:i+1], depth=-gt_depth[i:i+1]*1e3, foc_dist=-foc_dist*1e3, train=True)
                        focal_stack.append(defocus_img)
                        torch.cuda.empty_cache()
                    focal_stack = torch.cat(focal_stack,dim=0)
            torch.cuda.empty_cache()
            input_dict['stack_rgb_img'] = focal_stack

            optimizer.zero_grad()
            with autocast():
                losses, outputs = dfdp_net(input_dict)
                loss = losses['total'].mean()
                assert torch.isnan(loss).sum() == 0, print(loss)
            scaler.scale(loss).backward()
            clip_grad_norm_(parameters=dfdp_net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

@torch.no_grad()
def validate(net:Basenet, test_lens:PSFNet, valid_set, scene, args, epoch=0):
    valid_dataloader = DataLoader(valid_set, batch_size=1)
    num_val = len(valid_set)
    
    net.eval()
    result_img_dir = f'{args["results_dir"]}/results/'
    os.makedirs(result_img_dir, exist_ok=True)
    device = args['device']
    # torch.save(net.state_dict(), f'{args["results_dir"]}/depth_net_last.pkl')

    results_monitor = ResultsMonitor(args['train_mode'])    
    val_time = 0.0
    for idx, samples in enumerate(tqdm(valid_dataloader, desc="valid")):
        # Generate input
        aif, gt_depth = samples
        aif = aif.to(device)
        gt_depth = gt_depth.to(device)
        
        test_input_dict = {'gt_depth':gt_depth,'AiF_img':aif}
        # Render DoF image for input
        focus_dists = select_focus_dist(gt_depth, args['n_stack'], mode='linear')

        for i in range(args['n_stack']):    # n=1 in dfdp experiment
            foc_dist = focus_dists[:, i]
            dof_img = test_lens.render(aif, depth = - gt_depth * 1e3, foc_dist = - foc_dist * 1e3)
            focal_stack=dof_img

        torch.cuda.empty_cache()
        test_focal_stack = focal_stack
        test_input_dict['stack_rgb_img'] = test_focal_stack

        # Inference
        start = time.time()
        _, test_outputs = net.module.dfdp(test_input_dict)
        val_time = val_time + (time.time() - start)
        
        # AiF score matrics
        results_monitor.set_outputs(test_outputs)
        results_monitor.compute_metrics()
        results_monitor.save_images(result_img_dir,scene,idx)

    logging.info(f"Validate Depth Est on {scene}")
    results_monitor.logging(epoch, num_val)
    results_monitor.save_pth(args, scene, num_val, net)


@torch.no_grad()
def test(net:Basenet, test_set, scene, args, epoch=0):
    test_loader = DataLoader(test_set, batch_size=1)
    num_val = len(test_set)
    net.eval()
    result_img_dir = f'{args["results_dir"]}/tests/'
    os.makedirs(result_img_dir, exist_ok=True)
    device = args['device']
    
    # Score for depth prediction
    results_monitor = ResultsMonitor(args['train_mode']) 
    val_time = 0.0
    for idx, samples in enumerate(tqdm(test_loader, desc="valid")):
        
        # Generate input
        imgs, gt_depth = samples
        imgs = imgs.to(device)
        gt_depth = gt_depth.to(device)    # depth in [m]
    
        focal_stack = imgs
        torch.cuda.empty_cache()

        test_focal_stack = focal_stack # shape of [B, C, S, H, W]
        # Inference
        test_input_dict = {'stack_rgb_img': test_focal_stack, 'depth':gt_depth, 'AiF_img':None}
        
        start = time.time()            
        test_outputs = net.module.inference(test_input_dict)
        val_time = val_time + (time.time() - start)
        
        # AiF score matrics
        results_monitor.set_outputs(test_outputs)
        results_monitor.compute_metrics()
        results_monitor.save_images(result_img_dir,scene,idx)

    logging.info(f"Test Depth Est on {scene}")
    results_monitor.logging(epoch, num_val)
    results_monitor.save_pth(args, scene, num_val, net)

@torch.no_grad()
def test_DP_images(test_lens:PSFNet, flat_set, scene, args, epoch=0):
    flat_loader = DataLoader(flat_set, batch_size=1)
    num_val = len(flat_set)
    
    result_img_dir = f'{args["results_dir"]}/DPimages/'
    os.makedirs(result_img_dir, exist_ok=True)
    device = args['device']

    import xlsxwriter
    workbook = xlsxwriter.Workbook(f"{result_img_dir}/res.xlsx")
    worksheet = workbook.add_worksheet()
    first_item = ["idx", "distance", "psnr_l","psnr_r","ssim_l","ssim_r"]
    row = 0
    for col,item in enumerate(first_item):
        worksheet.write(row, col, item)
    psnr_ssim_record = []

    for idx, samples in enumerate(tqdm(flat_loader, desc="valid")):
        # Generate input
        f4_img, f20_img, depth = samples
        f4_img = f4_img.to(device)
        f20_img = f20_img.to(device)
        gt_depth = depth.to(device)

        focus_dists = select_focus_dist(gt_depth, args['n_stack'], mode='linear')  # n=1 in dfdp experiment

        f4_l, f4_r = f4_img[:,:3,:,:],f4_img[:,3:,:,:]
        f20_l, f20_r = f20_img[:,:3,:,:],f20_img[:,3:,:,:]
        for i in range(args['n_stack']):
            foc_dist = focus_dists[:, i]
            dof_img_l = test_lens.render(f20_l, depth = -gt_depth*1e3, foc_dist = -foc_dist*1e3)
            dof_img_l = dof_img_l[:,:3,:,:]
            torch.cuda.empty_cache()
            dof_img_r = test_lens.render(f20_r, depth = -gt_depth*1e3, foc_dist = -foc_dist*1e3)
            dof_img_r = dof_img_r[:,3:,:,:]
            torch.cuda.empty_cache()

        # AiF score matrics
        gt_aif_l = f4_l.detach().clone().cpu()
        pred_aif_l = dof_img_l.detach().clone().cpu()
        Avg_psnr_l = mask_psnr(pred_aif_l, gt_aif_l)
        Avg_ssim_l = mask_ssim(pred_aif_l, gt_aif_l)

        gt_aif_r = f4_r.detach().clone().cpu()
        pred_aif_r = dof_img_r.detach().clone().cpu()
        Avg_psnr_r = mask_psnr(pred_aif_r, gt_aif_r)
        Avg_ssim_r = mask_ssim(pred_aif_r, gt_aif_r)

        aif_l = f20_l.detach().clone().cpu()
        aif_r = f20_r.detach().clone().cpu()
        save_image(gt_aif_l, f'{result_img_dir}/img_{idx}_f4_real_l.png', normalize=False)
        save_image(aif_l, f'{result_img_dir}/img_{idx}_f20_real_l.png', normalize=False)
        save_image(pred_aif_l, f'{result_img_dir}/img_{idx}_f4_pred_l.png', normalize=False)
        save_image(gt_aif_r, f'{result_img_dir}/img_{idx}_f4_real_r.png', normalize=False)
        save_image(aif_r, f'{result_img_dir}/img_{idx}_f20_real_r.png', normalize=False)
        save_image(pred_aif_r, f'{result_img_dir}/img_{idx}_f4_pred_r.png', normalize=False)

        depth_mm = round((depth[0,0,0,0]*1e3).item())
        res = [idx, depth_mm, Avg_psnr_l, Avg_psnr_r, Avg_ssim_l, Avg_ssim_r]
        psnr_ssim_record.append(res)
        # print(f"[idx, depth (mm), psnr_l/psnr_r, ssiml/ssimr] : {res}]")
        row+=1
        for col,item in enumerate(res):
            worksheet.write(row, col, item)
        logging.info(f"[idx, depth (mm), psnr_l, psnr_r, ssim_l, ssim_r] : {res}")
    
    psnr_ssim_record_np = np.array(psnr_ssim_record)[:,2:]
    avg = np.average(psnr_ssim_record_np, axis=0)
    workbook.close()
    logging.info(f"Test DP Images on {scene}")
    logging.info(f"Avg [psnr_l, psnr_r, ssim_l, ssim_r]: {avg}")


if __name__=='__main__':
    args = config()
    train(args)
    destroy_process_group()