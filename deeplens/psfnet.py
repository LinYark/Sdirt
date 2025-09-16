""" Trainer for incoherent PSF network representation.
"""
import torch.nn.functional as nnF
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
from torch.cuda.amp import autocast, GradScaler

from .optics import *
from .render_psf import *
from .psfnet_arch import *
from .related_psf import generate_bw_kernel as bwk
from .related_psf import generate_modeling_kernel as mok
from .related_psf import generate_DPDNet_kernel as dpk

DMIN = 200     # [mm]
DMAX = 20000   # [mm]

class PSFNet(Lensgroup):
    def __init__(self, filename, model_name='mlp', kernel_size=11, sensor_res=(512, 512), device='cuda'):
        super(PSFNet, self).__init__(filename=filename, sensor_res=sensor_res, device=device)
        self.device = device

        # Init implicit PSF network
        self.in_features = 4
        self.kernel_size = kernel_size
        self.model_name = model_name
        self.init_net()

        # Training settings
        self.spp = 4096
        self.patch_size = 64
        self.psf_grid = [sensor_res[0] // self.patch_size, sensor_res[1] // self.patch_size]  # 512*512 sensor - 8*8 grid, 320*480 sensor - 5*8 grid, 480*640 sensor - 8*10 grid, 640*960 sensor - 10*15 grid
        
        # There is a minimum focal distance for each lens. 
        # For example, the Canon EF 50mm lens can only focus to 0.45m and further. 
        self.d_max = - DMAX
        self.d_min = - DMIN
        self.foc_d_arr = np.array([-500,   -600,    -700,    -800,    -900,    
                                   -1000,  -1250,   -1500,   -1750,   -2000, 
                                   -2500,  -3000,   -4000,   -5000,   -6000,   
                                   -8000,  -10000,  -12000,  -15000,  -20000])
        if filename.find("rf35mm") != -1:
            self.d_sensor = 80.447
        elif filename.find("rf50mm") != -1:
            self.d_sensor = 62.25
        else:
            print("filename is not correct")
            exit()

        self.foc_d_arr = np.array([-999.9,-1000,-1000.1], dtype=np.float32) + self.d_sensor
        self.foc_z_arr = (self.foc_d_arr - self.d_min) / (self.d_max - self.d_min)  # normalize focal distance [0, 1]
        self.foc_d = np.array([-1000.0], dtype=np.float32) + self.d_sensor

        # For others psf model
        self.psf_shot_modeling = None



    # ==================================================
    # Training functions
    # ==================================================
    def init_net(self):
        """ Initialize a network. 
        
            Basically there are three kinds of network architectures: (1) MLP, (2) MLP + Conv, (3) Siren. 
            
            We can also choose to represent (1) single-point PSF, (2) PSF map.

            Network input: (x, y, z, foc_dist), shape [N, 4].
            Network output: psf kernel (ks * ks) or psf map (psf_grid * ks * psf_grid * ks).
        """
        ks = self.kernel_size
        model_name = self.model_name

        if model_name == 'mlp':
            self.psfnet = MLP(in_features=3, out_features=ks**2, hidden_features=512, hidden_layers=8)
        elif model_name == 'mlpconv':
            self.psfnet = MLPConv(in_features=3, ks=ks, channels=1)
        elif model_name == 'mlp+lum':
            self.psfnet = MLP_lum(in_features=3, out_features=ks**2+1, hidden_features=512, hidden_layers=8)

        elif model_name == 'siren':
            raise NotImplementedError
        
        else:
            raise Exception('Unsupported PSF network architecture.')
        
        self.psfnet.apply(initialize_weights)
        self.psfnet.to(self.device)
    
    def load_net(self, net_path):
        """ Load pretrained network.
        """
        net_dict = self.psfnet.state_dict()
        pretrain_dict = torch.load(net_path)
        pretrained_dict = {k: v for k, v in pretrain_dict.items() if net_dict[k].shape == pretrain_dict[k].shape} 
        net_dict.update(pretrained_dict)
        self.psfnet.load_state_dict(net_dict)


    def train_psfnet(self, iters=10000, bs=128, lr=1e-4, spp=2048, evaluate_every=1000, result_dir='./results/temp'):
        """ Fit the PSF representation network. Training data is generated on the fly.
        """
        # Init network and prepare for training
        ks = self.kernel_size
        psfnet = self.psfnet
        psfnet.train()
        l2 = nn.MSELoss(reduction='mean')
        l1 = nn.L1Loss(reduction='mean')
        optim = torch.optim.AdamW(psfnet.parameters(), lr)
        sche = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(iters)//3, eta_min=0)
        scaler = GradScaler()

        # Train the network
        for i in tqdm(range(iters + 1),dynamic_ncols=True):
            # Training data
            with autocast():
                if self.model_name == 'mlp' or self.model_name == 'mlpconv':
                    inp, psf = self.get_training_data(bs=bs, spp=spp)
                else:
                    psf_grid = self.psf_grid
                    inp, psf = self.get_training_psf_map(bs=bs, psf_grid=psf_grid)
                inp, psf = inp.to(self.device), psf.to(self.device)

                psf_pred = psfnet(inp)
                loss_psf = l2(psf_pred, psf)
                optim.zero_grad()
            scaler.scale(loss_psf).backward()
            scaler.step(optim)
            scaler.update()
            sche.step()

            # Evaluate
            with torch.no_grad():
                with autocast():
                    if (i+1) % evaluate_every == 0:
                        psfnet.eval()

                        plot_num = 5
                        fig, axs = plt.subplots(plot_num, 2)
                        for j in range(plot_num):
                            psf0 = psf[j, :].detach().clone().cpu()
                            psf0 = psf0.view(1, ks, ks)
                            axs[j, 0].imshow(psf0[0])
                            # axs[j, 1].imshow(psf0[0])
                            psf1 = psf_pred[j, :].detach().clone().cpu()
                            psf1 = psf1.view(1, ks, ks)
                            axs[j, 1].imshow(psf1[0])
                            # axs[j, 3].imshow(psf1[0])
                        fig.suptitle(f"GT/Pred PSFs at iter {i+1}")
                        plt.savefig(f'{result_dir}/iter{i+1}.png', dpi=300)
                        plt.close()
                        torch.save(psfnet.state_dict(), f'{result_dir}/iter{i+1}_PSFNet_{self.model_name}.pkl')

                        inp, psf = self.get_test_data()
                        inp, psf = inp.to(self.device), psf.to(self.device)
                        psf_pred = psfnet(inp)
                        
                        psf = psf / psf.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
                        psf_pred = psf_pred / psf_pred.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
                        l1_loss = l1(psf_pred, psf)
                        l2_loss = l2(psf_pred, psf)
                        loss_print = f"{i}, {l1_loss.item()}, {l2_loss.item()}"
                        logging.info(loss_print)

                        psfnet.train()
        torch.save(psfnet.state_dict(), f'{result_dir}/PSFNet_{self.model_name}.pkl')


    def get_training_data(self, bs=256, spp=4096):
        """ Generate training data for a focus distance (f_d) and a group of spatial points (x, y, z).

            Input (x, y, z, foc_dist) range from [-1, 1] * [-1, 1] * [0, 1]
            Output (psf) normalized to 1D tensor.
        
        Args:
            bs (int): batch size
            spp (int): ray samples per pixel
        """
        # Sample only one f_d and refocus the lens
        foc_z = np.random.choice(self.foc_z_arr)
        foc_dist = foc_z * (self.d_max - self.d_min) + self.d_min
        # self.refocus(depth = foc_dist)
        
        # Sample (x, y), uniform distribution
        x = (torch.rand(bs) - 0.5) * 2
        y = (torch.rand(bs) - 0.5) * 2

        # Sample (z), Gaussian distribution (3-sigma interval)
        z_gauss = torch.clamp(torch.randn(bs), min=-3, max=3)
        z = torch.zeros_like(z_gauss)
        z[z_gauss>0] = (1 - foc_z) * z_gauss[z_gauss>0]/3 + foc_z   # sample [foc_z, 1], then scale to [foc_d, dmax]
        z[z_gauss<0] = foc_z * z_gauss[z_gauss<0]/3 + foc_z         # sample [0, foc_z], then scale to [dmin, foc_d]

        # Network input, shape of [N, 4]
        inp = torch.stack((x, y, z), dim=-1)

        # Ray tracing to compute PSFs, shape of [N, ks, ks]
        depth = self.z2depth(z)
        points = torch.stack((x, y, depth), dim=-1)
        psf = self.psf(points=points, ks=self.kernel_size, spp=spp)
        return inp, psf
    
    def get_test_data(self, bs=1024, spp=65536):
        """ Generate training data for a focus distance (f_d) and a group of spatial points (x, y, z).

            Input (x, y, z, foc_dist) range from [-1, 1] * [-1, 1] * [0, 1]
            Output (psf) normalized to 1D tensor.
        
        Args:
            bs (int): batch size
            spp (int): ray samples per pixel
        """
        # Sample only one f_d and refocus the lens
        foc_z = self.foc_z_arr[1]
        foc_dist = foc_z * (self.d_max - self.d_min) + self.d_min
        # self.refocus(depth = foc_dist)
        
        # Sample (x, y), uniform distribution
        psf_grid = 32
        x, y = torch.meshgrid(
            torch.linspace(-1 + 1/(2*psf_grid), 1 - 1/(2*psf_grid), psf_grid),
            torch.linspace(1 - 1/(2*psf_grid), -1 + 1/(2*psf_grid), psf_grid),
            indexing='xy'
        )
        x, y = x.reshape(-1), y.reshape(-1)

        # Sample (z), Gaussian distribution (3-sigma interval)
        z_gauss = torch.linspace(-3, 3, bs)
        z = torch.zeros_like(z_gauss)
        z[z_gauss>0] = (1 - foc_z) * z_gauss[z_gauss>0]/3 + foc_z   # sample [foc_z, 1], then scale to [foc_d, dmax]
        z[z_gauss<0] = foc_z * z_gauss[z_gauss<0]/3 + foc_z         # sample [0, foc_z], then scale to [dmin, foc_d]

        # Network input, shape of [N, 4]
        inp = torch.stack((x, y, z), dim=-1)

        # Ray tracing to compute PSFs, shape of [N, ks, ks]
        depth = self.z2depth(z)
        points = torch.stack((x, y, depth), dim=-1)
        psf = self.psf(points=points, ks=self.kernel_size, spp=spp)
        return inp, psf
    
    def get_training_psf_map(self, bs=8, psf_grid=(11,11), psf_map_size=(128, 128)):
        """ Generate PSF map for training. This training data is used for MLP_Conv network architecture.

            Reference: "Differentiable Compound Optics and Processing Pipeline Optimization for End-To-end Camera Design."

        Args:
            bs (int): batch size
            psf_grid (tuple): PSF grid size
            psf_map_size (tuple): PSF map size

        Returns:
            inp (tensor): [B, 2] tensor, [z, foc_z]
            psf_map_batch (tensor): [B, 3, psf_map_size, psf_map_size] tensor
        """
        # Refocus
        foc_z = np.random.choice(self.foc_z_arr)
        foc_dist = foc_z * (self.d_max - self.d_min) + self.d_min

        # Different depths
        z_gauss = torch.clamp(torch.randn(bs), min=-3, max=3)
        z = torch.zeros_like(z_gauss)
        z[z_gauss>0] = (1 - foc_z) * z_gauss[z_gauss>0]/3 + foc_z
        z[z_gauss<0] = foc_z * z_gauss[z_gauss<0]/3 + foc_z
        depth = self.z2depth(z)

        # 2D Input (foc_z, z)
        foc_z_tensor = torch.full_like(z, foc_z)
        inp = torch.stack((z, foc_z_tensor), dim=-1)    # [B, 2]

        # Calculate PSF map
        psf_map_batch = []
        for depth_i in depth:
            psf_map = self.calc_psf_map(foc_dist, depth_i, psf_grid=psf_grid)
            psf_map_batch.append(psf_map)
        psf_map_batch = torch.stack(psf_map_batch, dim=0)   # [B, 3, psf_grid*ks, psf_grid*ks]
        
        # Resize to meet the network requirement
        psf_map_batch = F.resize(psf_map_batch, psf_map_size)   # [B, 3, size, size]

        return inp, psf_map_batch


    def calc_psf_map(self, foc_dist, depth, psf_grid=(11, 11)):
        """ Calculate PSF grid by ray tracing. 

            This function is similiar for self.psf() function.
        """
        ks = self.kernel_size
        spp = self.spp

        # Focus to given distance
        self.refocus(depth = foc_dist)

        # Sample grid points
        x, y = torch.meshgrid(
            torch.linspace(-1 + 1/(2*psf_grid[1]), 1 - 1/(2*psf_grid[1]), psf_grid[1]),
            torch.linspace(1 - 1/(2*psf_grid[0]), -1 + 1/(2*psf_grid[0]), psf_grid[0]),
            indexing='xy'
        )
        x, y = x.reshape(-1), y.reshape(-1)
        depth = torch.full_like(x, depth)
        o = torch.stack((x, y, depth), dim=-1)
        
        # Calculate PSf by ray-tracing
        psf = self.psf(points=o, kernel_size=ks, spp=spp, center=True) # [psf_grid^2, ks, ks]

        # Convert to tensor and save image
        psf_map = make_grid(psf.unsqueeze(1), nrow = psf_grid[1], padding=0)    # [3, psf_grid*ks, psf_grid*ks]
        
        return psf_map

    # ==================================================
    # Use network after image simulation
    # ==================================================
    def pred(self, inp):
        """ Predict PSFs using the PSF network.

        Args:
            inp (tensor): [N, 3] tensor, [x, y, z]

        Returns:
            psf (tensor): [N, ks, ks] or [H, W, ks, ks] tensor
        """
        # Network prediction, shape of [N, ks^2]
        psfl = self.psfnet(inp)
        inp[...,0] = inp[...,0]*(-1)
        psfr = self.psfnet(inp)
        psfr = torch.flip(psfr,dims=[-1])

        psf = torch.stack((psfl,psfr), dim=-3)
        psf = psf / (psf.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1) + 1e-9)

        assert psf.shape[-1] == self.kernel_size
        return psf
        
    def pred_coc(self, inp, is_z=True):
        ks = self.kernel_size
        x_gaussi, y_gaussi = torch.meshgrid(
            torch.linspace(-ks/2+1/2, ks/2-1/2, ks).to(self.device),
            torch.linspace(-ks/2+1/2, ks/2-1/2, ks).to(self.device),
            indexing='xy'
        )

        # inp: B,W,H,3
        z = inp[...,-1]
        B,W,H= inp.shape[0:3]
        foc_dist = torch.tensor(self.foc_d).to(self.device)
        foc_dist_z = self.depth2z(foc_dist)
        ps = self.sensor_size[0] / self.sensor_res[0]

        depth = z * (self.d_max - self.d_min) + self.d_min if is_z else z
        coc = torch.abs(depth - foc_dist) * self.foclen**2 / (-depth * self.fnum *(-foc_dist - self.foclen))
        coc_pixel = torch.clamp(coc / ps, min=0.1)
        coc_pixel_radius = coc_pixel / 2
        coc_pixel_radius = coc_pixel_radius.unsqueeze(-1).unsqueeze(-1)
        x_gaussi_, y_gaussi_ = x_gaussi.repeat(B,W,H,1,1), y_gaussi.repeat(B,W,H,1,1)
        psf_thin = torch.exp(- (x_gaussi_**2 + y_gaussi_**2) / (2 * coc_pixel_radius**2)) # We ignore constant term because PSF will be normalized later
        psf_mask = (x_gaussi_**2 + y_gaussi_**2 < coc_pixel_radius**2)
        psf_thin = psf_thin * psf_mask # Un-clipped Gaussian PSF
        l_mask,r_mask = torch.ones((ks,ks)).to(self.device).repeat(1,1,1,1,1),torch.ones((ks,ks)).to(self.device).repeat(1,1,1,1,1)
        l_pixel, r_pixel = ks//2, ks//2+1
        near_focus_pos,far_focus_pos = torch.where(depth > foc_dist),  torch.where(depth < foc_dist)
        l_mask[...,0:l_pixel] = l_mask[...,0:l_pixel]*0
        r_mask[...,r_pixel:] = r_mask[...,r_pixel:]*0

        psf_l, psf_r = psf_thin*1, psf_thin*1
        psf_l[near_focus_pos] = psf_l[near_focus_pos] * l_mask
        psf_l[far_focus_pos] = psf_l[far_focus_pos] * r_mask
        psf_r[near_focus_pos] = psf_r[near_focus_pos] * r_mask
        psf_r[far_focus_pos] = psf_r[far_focus_pos] * l_mask
        psf = torch.stack((psf_l, psf_r), dim=-3)
        psf = psf / (psf.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1) + 1e-6)

        return psf
    
    def pred_DPDNet(self, inp, is_z=True):
        B,W,H,_ = inp.shape
        ks = self.kernel_size
        radius_step = 0.5
        radius_range = torch.arange(radius_step,ks,step=radius_step)  #radius
        if self.psf_shot_modeling==None:
            psf_list = []
            for i_radius in radius_range:
                psf_l, psf_r = dpk.ker_rect(i_radius, ks)
                psf = np.stack([psf_l, psf_r],axis=0)
                psf_list.append(psf)
            psf_np = np.stack(psf_list,axis=0)
            psf_shot = torch.tensor(psf_np, dtype=torch.float32).to(self.device)
            self.psf_shot_modeling = psf_shot

        z = inp[...,-1]
        depth = z * (self.d_max - self.d_min) + self.d_min if is_z else z
        foc_dist = torch.tensor(self.foc_d).to(self.device)
        coc_sign = (depth - foc_dist) * self.foclen**2 / (-depth * self.fnum *(-foc_dist - self.foclen))
        coc = torch.abs(coc_sign)
        ps = self.sensor_size[0] / self.sensor_res[0]
        coc_pixel_radius = torch.clamp(coc / ps /2.0, min=0.1)
        coc_pixel_radius = np.sqrt(torch.pi) * coc_pixel_radius /2.0
        psfmap_mod_l = torch.ones([B,W,H,ks,ks],dtype=torch.float32).to(self.device)
        psfmap_mod_r = torch.ones([B,W,H,ks,ks],dtype=torch.float32).to(self.device)

        for i_radius in radius_range:
            range_l = i_radius-radius_step
            range_r = i_radius

            idx_shot = range_l/radius_step
            psf_patch_l, psf_patch_r = self.psf_shot_modeling[idx_shot.int().item()][0], self.psf_shot_modeling[idx_shot.int().item()][1]

            psfmap_mod_l[torch.logical_and(torch.logical_and(coc_pixel_radius>=range_l, coc_pixel_radius<range_r), coc_sign>=0)] = psf_patch_l
            psfmap_mod_r[torch.logical_and(torch.logical_and(coc_pixel_radius>=range_l, coc_pixel_radius<range_r), coc_sign>=0)] = psf_patch_r
            psfmap_mod_l[torch.logical_and(torch.logical_and(coc_pixel_radius>=range_l, coc_pixel_radius<range_r), coc_sign<0)] = psf_patch_r
            psfmap_mod_r[torch.logical_and(torch.logical_and(coc_pixel_radius>=range_l, coc_pixel_radius<range_r), coc_sign<0)] = psf_patch_l
            if i_radius==ks-radius_step:
                psfmap_mod_l[torch.logical_and(coc_pixel_radius>=range_r, coc_sign>=0)] = psf_patch_l
                psfmap_mod_r[torch.logical_and(coc_pixel_radius>=range_r, coc_sign>=0)] = psf_patch_r
                psfmap_mod_l[torch.logical_and(coc_pixel_radius>=range_r, coc_sign<0)] = psf_patch_r
                psfmap_mod_r[torch.logical_and(coc_pixel_radius>=range_r, coc_sign<0)] = psf_patch_l

        psf = torch.stack((psfmap_mod_r,psfmap_mod_l), dim=-3)
        psf = psf / (psf.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1) + 1e-6)
        return psf

    def pred_Modeling(self, inp, is_z=True):
        B,W,H,_ = inp.shape
        ks = self.kernel_size
        radius_step = 0.5
        radius_range = torch.arange(radius_step,ks,step=radius_step)  #radius
        if self.psf_shot_modeling==None:
            psf_list = []
            for i_radius in radius_range:
                psf_l, psf_r = mok.ker_disk(i_radius, ks)
                psf = np.stack([psf_l, psf_r],axis=0)
                psf_list.append(psf)
            psf_np = np.stack(psf_list,axis=0)
            psf_shot = torch.tensor(psf_np, dtype=torch.float32).to(self.device)
            self.psf_shot_modeling = psf_shot

        z = inp[...,-1]
        depth = z * (self.d_max - self.d_min) + self.d_min if is_z else z
        foc_dist = torch.tensor(self.foc_d).to(self.device)
        coc_sign = (depth - foc_dist) * self.foclen**2 / (-depth * self.fnum *(-foc_dist - self.foclen))
        coc = torch.abs(coc_sign)
        ps = self.sensor_size[0] / self.sensor_res[0]
        coc_pixel_radius = torch.clamp(coc / ps /2.0, min=0.1)
        psfmap_mod_l = torch.ones([B,W,H,ks,ks],dtype=torch.float32).to(self.device)
        psfmap_mod_r = torch.ones([B,W,H,ks,ks],dtype=torch.float32).to(self.device)

        for i_radius in radius_range:
            range_l = i_radius-radius_step
            range_r = i_radius

            idx_shot = range_l/radius_step
            psf_patch_l, psf_patch_r = self.psf_shot_modeling[idx_shot.int().item()][0], self.psf_shot_modeling[idx_shot.int().item()][1]

            psfmap_mod_l[torch.logical_and(torch.logical_and(coc_pixel_radius>=range_l, coc_pixel_radius<range_r), coc_sign>=0)] = psf_patch_l
            psfmap_mod_r[torch.logical_and(torch.logical_and(coc_pixel_radius>=range_l, coc_pixel_radius<range_r), coc_sign>=0)] = psf_patch_r
            psfmap_mod_l[torch.logical_and(torch.logical_and(coc_pixel_radius>=range_l, coc_pixel_radius<range_r), coc_sign<0)] = psf_patch_r
            psfmap_mod_r[torch.logical_and(torch.logical_and(coc_pixel_radius>=range_l, coc_pixel_radius<range_r), coc_sign<0)] = psf_patch_l
            if i_radius==ks-radius_step:
                psfmap_mod_l[torch.logical_and(coc_pixel_radius>=range_r, coc_sign>=0)] = psf_patch_l
                psfmap_mod_r[torch.logical_and(coc_pixel_radius>=range_r, coc_sign>=0)] = psf_patch_r
                psfmap_mod_l[torch.logical_and(coc_pixel_radius>=range_r, coc_sign<0)] = psf_patch_r
                psfmap_mod_r[torch.logical_and(coc_pixel_radius>=range_r, coc_sign<0)] = psf_patch_l

        psf = torch.stack((psfmap_mod_l, psfmap_mod_r), dim=-3)
        psf = psf / (psf.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1) + 1e-6)
        return psf
    
    def pred_Learn2reduce(self, inp, is_z=True):
        B,W,H,_ = inp.shape
        ks = self.kernel_size
        # inp: B,W,H,3
        z = inp[...,-1]
        depth = z * (self.d_max - self.d_min) + self.d_min if is_z else z
        foc_dist = torch.tensor(self.foc_d).to(self.device)
        coc_sign = (depth - foc_dist) * self.foclen**2 / (-depth * self.fnum *(-foc_dist - self.foclen))
        coc = torch.abs(coc_sign)
        ps = self.sensor_size[0] / self.sensor_res[0]
        coc_pixel = torch.clamp(coc / ps, min=0.1)
        coc_pixel = (coc_pixel//2*2+1).int()
        psfmap_L2R_l = torch.ones([B,W,H,ks,ks],dtype=torch.float32).to(self.device)
        psfmap_L2R_r = torch.ones([B,W,H,ks,ks],dtype=torch.float32).to(self.device)

        def psf_crop(kernel,psf_size=21):
            psf=np.zeros([psf_size,psf_size])
            ks = kernel.shape[0]//2
            ps = psf_size//2

            range_l = np.abs(ks-ps)
            range_r = np.abs(ks+ps)+1
            if ks>=ps:
                psf=kernel[range_l:range_r,range_l:range_r]
            else:
                psf[range_l:range_r,range_l:range_r]=kernel
            psf/=psf.sum()
            psf_tensor = torch.tensor(psf,dtype=torch.float32).to(self.device)
            return psf_tensor
        
        coc_list = torch.arange(1,ks*2+1,2)  # radius*2
        for i_pixel in coc_list:
            if i_pixel == 1:
                psf_patch_l = torch.zeros([ks,ks]).to(self.device)
                psf_patch_r = torch.zeros([ks,ks]).to(self.device)
                psf_patch_l[ks//2,ks//2]=1
                psf_patch_r[ks//2,ks//2]=1
            else:
                # order, cut_off_factor, beta = 6, 0.4, 0.4
                order, cut_off_factor, beta = 3, 2.5, 0.2
                smooth_strength = 3
                kernel_c, kernel_r, kernel_l = bwk.bw_kernel_generator(i_pixel.item(), order, cut_off_factor, beta, smooth_strength)
                psf_patch_r, psf_patch_l = psf_crop(kernel_r,ks), psf_crop(kernel_l,ks)

            psfmap_L2R_l[torch.logical_and(coc_pixel==i_pixel, coc_sign>=0)] = psf_patch_l
            psfmap_L2R_r[torch.logical_and(coc_pixel==i_pixel, coc_sign>=0)] = psf_patch_r
            psfmap_L2R_l[torch.logical_and(coc_pixel==i_pixel, coc_sign<0)] = psf_patch_r
            psfmap_L2R_r[torch.logical_and(coc_pixel==i_pixel, coc_sign<0)] = psf_patch_l
            if i_pixel==ks*2-1:
                psfmap_L2R_l[torch.logical_and(coc_pixel>=i_pixel, coc_sign>=0)] = psf_patch_l
                psfmap_L2R_r[torch.logical_and(coc_pixel>=i_pixel, coc_sign>=0)] = psf_patch_r
                psfmap_L2R_l[torch.logical_and(coc_pixel>=i_pixel, coc_sign<0)] = psf_patch_r
                psfmap_L2R_r[torch.logical_and(coc_pixel>=i_pixel, coc_sign<0)] = psf_patch_l

        psf = torch.stack((psfmap_L2R_r, psfmap_L2R_l), dim=-3)
        psf = psf / (psf.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1) + 1e-6)
        return psf

    def compare_psf(self):
        x = torch.Tensor([0, 0.4, 0.8])
        y = torch.Tensor([0, 0.4, 0.8])
        test_dists = torch.Tensor([-500, -20000])

        for depth in test_dists:
            d_ori = depth
            depth = depth+self.d_sensor

            d_tensor = torch.full_like(x, depth)
            inp = torch.stack((x, y, d_tensor), dim=-1) # [3(stack),3]
            psfl = self.psf(points=inp, ks=self.kernel_size, center=True,spp=GEO_SPP*100).to("cpu")

            inp[...,0] = inp[...,0]*(-1)
            psfr = self.psf(points=inp, ks=self.kernel_size, center=True,spp=GEO_SPP*100).to("cpu")
            psfr = torch.flip(psfr,dims=[-1])

            psf = torch.stack([psfl[0], psfr[0]],dim=0)
            self.vis_psf_map(psf, filename=f'./rt_{int(d_ori)}_v00.png')
            psf = torch.stack([psfl[1], psfr[1]],dim=0)
            self.vis_psf_map(psf, filename=f'./rt_{int(d_ori)}_v04.png')
            psf = torch.stack([psfl[2], psfr[2]],dim=0)
            self.vis_psf_map(psf, filename=f'./rt_{int(d_ori)}_v08.png')

            z = self.depth2z(depth)
            z_tensor = torch.full_like(x, z)
            inp = torch.stack((x, y, z_tensor), dim=-1)
            inp = inp.repeat(1,1,1,1).to(self.device) 
            psf2 = self.pred(inp).detach().to("cpu")
            # psf2 = self.pred_coc(inp).detach().to("cpu") # inp = b w h 3
            # psf2 = self.pred_DPDNet(inp).detach().to("cpu") # inp = b w h 3
            # psf2 = self.pred_Modeling(inp).detach().to("cpu") # inp = b w h 3
            # psf2 = self.pred_Learn2reduce(inp).detach().to("cpu") # inp = b w h 3
            psfstack = torch.stack([psf2[0,0,0,0,...],psf2[0,0,0,1,...]],dim=0)
            self.vis_psf_map(psfstack, filename=f'./pred_{int(d_ori)}_v00.png')
            psfstack = torch.stack([psf2[0,0,1,0,...],psf2[0,0,1,1,...]],dim=0)
            self.vis_psf_map(psfstack, filename=f'./pred_{int(d_ori)}_v04.png')
            psfstack = torch.stack([psf2[0,0,2,0,...],psf2[0,0,2,1,...]],dim=0)
            self.vis_psf_map(psfstack, filename=f'./pred_{int(d_ori)}_v08.png')
    
    
    def time_compare_psf(self):
        import time

        inp = torch.rand(512*768//16,3)
        inp[:,2] = self.z2depth(inp[:,2])
        start_time = time.time()
        psfl = self.psf(points=inp, ks=self.kernel_size, center=True,spp=GEO_SPP*2).to("cpu")
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"ray_tracing time cost: {execution_time}s")

        inp = torch.rand(1,512//4,768//4,3).cuda()
        start_time = time.time()
        psf2 = self.pred(inp).detach().to("cpu")
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"network time cost: {execution_time}s")


    def fit_degamma(self,x):
        #255->1k, l for lum
        a1,b1,c1=0.89129432,  0.27217316, -0.00246187
        a2,b2,c2=5.94018909e-01,  1.20060450e+01, -5.24983855e-03
        l1 = 1/(1/(a1*x+b1)+c1)
        l2 = 1/(1/(a2*x+b2)+c2)
        ratio_x = x/100
        ratio_x[ratio_x>1] = 1
        l = l2*ratio_x + l1*(1-ratio_x)
        return l
    
    def degamma(self,img_gamma):
        img_255 = img_gamma*255.
        img_degamma = self.fit_degamma(img_255)
        return img_degamma

    def fit_gamma(self,l):
        #1k->255, l for lum
        a1,b1,c1=0.89129432,  0.27217316, -0.00246187
        a2,b2,c2=5.94018909e-01,  1.20060450e+01, -5.24983855e-03
        x1 =(1/(1/(l+1e-9)-c1)-b1)/a1
        x2 =(1/(1/(l+1e-9)-c2)-b2)/a2
        xmid = (x1+x2)/2
        ratio_x = xmid/100
        ratio_x[ratio_x>1] = 1
        x = x2*ratio_x + x1*(1-ratio_x)
        return x

    def gamma(self,img_degamma):
        img_255 = self.fit_gamma(img_degamma)
        img_gamma = img_255/255.
        return img_gamma
    
    
    def exp_response(self, render_img,  val=False):
        render_img_255 = render_img*255.
        render_img_exp = self.final_r(render_img_255, val)
        render_img_norm = render_img_exp/255.
        return render_img_norm
    
    def noise(self,render, shape):
        N, C, H, W = shape
        noise_range = 0.05*np.random.rand()
        noise_map = torch.randn_like(render) * noise_range
        range1, range2 = (np.random.rand()/2), (np.random.rand()/2+0.5)
        weight_l = torch.linspace(range1,range2,W)
        weight_l = weight_l.repeat(N,C,H,1)
        weight_r = torch.flip(weight_l,[-1])
        weight_map = torch.cat([weight_l,weight_r],dim=1).to(self.device)
        noise = noise_map*weight_map
        render +=  noise
        # noise=0.03*np.random.rand()
        # render += torch.randn_like(render) * noise
        return render

    @torch.no_grad()
    def render(self, img, depth, foc_dist, train=False):
        """ Render image with aif image and depth map. Receive [N, C, H, W] image.

        Args:
            img (tensor): [N, C, H, W]
            depth (tensor): [N, H, W], depth map, unit in mm, range from [-20000, -200]
            foc_dist (tensor): [N], unit in mm, range from [-20000, -200]
            high_res (bool): whether to use high resolution rendering

        Returns:
            render (tensor): [N, C, H, W]
        """
        # fix ignoring d_sensor bug
        depth = depth+self.d_sensor
        foc_dist = foc_dist+self.d_sensor
        
        if len(img.shape) == 3:
            H, W = depth.shape
            z = self.depth2z(depth)
            x, y = torch.meshgrid(
                torch.linspace(-1,1,W),
                torch.linspace(1,-1,H),
                indexing='xy'
            )
            x, y = x.to(self.device), y.to(self.device)
            foc_dist = torch.full_like(depth, foc_dist)
            foc_z = self.depth2z(foc_dist)
            
            o = torch.stack((x, y, z, foc_z), -1)
            psf = self.pred(o)

            render_lr = local_psf_render(img, psf, self.kernel_size)
            render = torch.cat(render_lr, dim=1)
            return render
        
        elif len(img.shape) == 4:
            N, C, H, W = img.shape
            # depth = (torch.ones_like(depth).to(self.device))*(-500)
            z = self.depth2z(depth).squeeze(1)
            x, y = torch.meshgrid(
                torch.linspace(-1,1,W),
                torch.linspace(1,-1,H),
                indexing='xy'
            )
            x, y = x.unsqueeze(0).repeat(N,1,1), y.unsqueeze(0).repeat(N,1,1)
            x, y = x.to(img.device), y.to(img.device)
            foc_dist = foc_dist.unsqueeze(-1).unsqueeze(-1).repeat(1,H,W)
            foc_z = self.depth2z(foc_dist)

            o = torch.stack((x, y, z), -1).float()

            # Other models
            # psf = self.pred_coc(o)
            # # psf = self.pred_DPDNet(o)
            # # psf = self.pred_Modeling(o)
            # # psf = self.pred_Learn2reduce(o)
            # render_lr = local_psf_render_fast(img, psf, self.kernel_size)
            # render = torch.cat(render_lr, dim=1)

            # Our Sdirt
            psf = self.pred(o)  
            img_degamma = self.degamma(img)
            render_lr = local_psf_render_fast(img_degamma, psf, self.kernel_size)
            render = torch.cat(render_lr, dim=1)
            render = self.gamma(render)

            if train ==True:
                render = self.noise(render, img.shape)
            render = torch.clip(render, 0.0, 1.0)
            return render

    # ==================================================
    # Utils
    # ==================================================
    def depth2z(self, depth):
        z = (depth - self.d_min) / (self.d_max - self.d_min)
        z = torch.clamp(z, min=0, max=1)
        return z

    def z2depth(self, z):
        depth = z * (self.d_max - self.d_min) + self.d_min
        return depth
    
    def vis_psf_map(self, psf, filename=None, normal=True):
        """ Visualize a [N, N, k, k] or [N, N, k^2] or [N, k, k] PSF kernel.
        """
        if len(psf.shape) == 4:
            N, C, _, _ = psf.shape
            if N == C:
                fig, axs = plt.subplots(N, N)
                for i in range(N):
                    for j in range(N):
                        psf0 = psf[i, j, :, :].detach().clone().cpu()
                        axs[i, j].imshow(psf0, vmin=0.0, vmax=0.1)
            else:
                fig, axs = plt.subplots(C, N)
                for i in range(C):
                    for j in range(N):
                        psf0 = psf[j, i, :, :].detach().clone().cpu()
                        axs[i, j].imshow(psf0, vmin=0.0, vmax=0.1)

        elif len(psf.shape) == 3:
            N, _, _ = psf.shape
            fig, axs = plt.subplots(1, N)
            for i in range(N):
                psf0 = psf[i, :, :].detach().clone().cpu()
                if normal:
                    psf0 /= psf0.max()
                axs[i].imshow(psf0, vmin=0.0, vmax=1, cmap='gray')
                axs[i].axis('off')

        # Return fig
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

        plt.close()

    def vis_psf_map_np(self, psf, filename=None):
        """ Visualize a [N, N, k, k] or [N, N, k^2] or [N, k, k] PSF kernel.
        """

        N, _, _ = psf.shape
        fig, axs = plt.subplots(1, N)
        for i in range(N):
            psf0 = psf[i, :, :]
            psf0 /= psf0.max()
            axs[i].imshow(psf0, vmin=0.0, vmax=1, cmap='gray')
            axs[i].axis('off')

        # Return fig
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

        plt.cla()
   

# ==================================================================
# Thin lens model (baseline)
# ==================================================================
class ThinLens(DeepObj):
    def __init__(self, foc_len, fnum, kernel_size, sensor_size, sensor_res, device='cpu'):
        super(ThinLens, self).__init__()

        self.d_max = DMAX
        self.d_min = DMIN
        self.kernel_size = kernel_size
        self.foc_len = foc_len
        self.fnum = fnum
        self.sensor_size = sensor_size
        self.sensor_res = sensor_res
        self.ps = self.sensor_size[0] / self.sensor_res[0]


    def coc(self, depth, foc_dist):
        if (depth < 0).any():
            depth = - depth
            foc_dist = - foc_dist

        depth = torch.clamp(depth, self.d_min, self.d_max)
        coc = self.foc_len / self.fnum * torch.abs(depth - foc_dist) / depth * self.foc_len / (foc_dist - self.foc_len)
        coc_pixel = torch.clamp(coc / self.ps, min=0.1)
        return coc_pixel


    def render(self, img, depth, foc_dist):
        """ Render image with aif image and Gaussian PSFs.

        Args:
            img: [N, C, H, W]
            depth: [N, 1, H, W]
            foc_dist: [N]

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """
        ks = self.kernel_size
        device = img.device
        
        if len(img.shape) == 3:
            H, W = depth.shape
            z = self.depth2z(depth)
            x, y = torch.meshgrid(
                torch.linspace(-1,1,W),
                torch.linspace(1,-1,H),
                indexing='xy'
            )
            x, y = x.to(self.device), y.to(self.device)
            foc_dist = torch.full_like(depth, foc_dist)
            foc_z = self.depth2z(foc_dist)
            o = torch.stack((x, y, z, foc_z), -1)    
            
            psf = self.pred(o)
            render = local_psf_render(img, psf, self.kernel_size)
            
            return render
        
        elif len(img.shape) == 4:
            N, C, H, W = img.shape
            foc_dist = foc_dist.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W) # [N] to [N, 1, H, W]
            psf = torch.zeros((N, H, W, ks, ks), device=device)
            x, y = torch.meshgrid(
                torch.linspace(-ks/2+1/2, ks/2-1/2, ks),
                torch.linspace(ks/2-1/2, -ks/2+1/2, ks),
                indexing='xy'
            )
            x, y = x.to(device), y.to(device)

            coc_pixel = self.coc(depth, foc_dist)
            # Shape expands to [N, H, W, ks, ks]
            coc_pixel = coc_pixel.squeeze(1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, ks, ks)
            coc_pixel_radius = coc_pixel / 2
            psf = torch.exp(- (x**2 + y**2) / 2 / coc_pixel_radius**2) / (2 * np.pi * coc_pixel_radius**2)
            psf_mask = (x**2 + y**2 < coc_pixel_radius**2)
            psf = psf * psf_mask
            psf = psf / psf.sum((-1, -2)).unsqueeze(-1).unsqueeze(-1)

            render = local_psf_render(img, psf, self.kernel_size)
            return render