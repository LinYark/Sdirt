"""
Render sensor image by PSF convolution.
"""
import numpy as np
import torch
import torch.nn.functional as nnF


# ================================================
# PSF convolution
# ================================================
def render_psf(img, psf):
    """ Render an image with PSF. Use the same PSF kernel for all pixels.  

    Args:
        img (torch.Tensor): [B, C, H, W]
        psf (torch.Tensor): [C, ks, ks]
    
    Returns:
        img_render (torch.Tensor): [B, C, H, W]
    """
    _, ks, ks = psf.shape
    padding = int(ks / 2)
    psf = torch.flip(psf, [1, 2])  # flip the PSF because nnF.conv2d use cross-correlation
    psf = psf.unsqueeze(1)  # shape [C, 1, ks, ks]
    img_pad = nnF.pad(img, (padding, padding, padding, padding), mode='reflect')
    img_render = nnF.conv2d(img_pad, psf, groups=img.shape[1], padding=0, bias=None)
    return img_render


def render_psf_map(img, psf_map, grid):
    """ Render an image with PSF map. Use the different PSF kernel for different patches.

        Args:
            img (torch.Tensor): [B, 3, H, W]
            psf_map (torch.Tensor): [3, grid*ks, grid*ks]
            grid (int): grid number

        Returns:
            render_img (torch.Tensor): [B, C, H, W]
    """
    if torch.is_tensor(img):
        assert len(img.shape) == 4, 'Input image should be [B, C, H, W]'
    else:
        img = torch.tensor((img/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)

    Cpsf, Hpsf, Wpsf = psf_map.shape
    assert Hpsf % grid == 0 and Wpsf % grid == 0, 'PSF map size should be divisible by grid'
    ks = int(Hpsf / grid)
    assert ks % 2 == 1, 'PSF kernel size should be odd'

    B, C, H, W = img.shape
    assert C == Cpsf, 'PSF map should have the same channel as image'
    
    pad = int((ks-1)/2)
    patch_size = int(H/grid)
    img_pad = nnF.pad(img, (pad, pad, pad, pad), mode='reflect')
    
    render_img = torch.zeros_like(img)
    for i in range(grid):
        for j in range(grid):
            psf = psf_map[:, i*ks:(i+1)*ks, j*ks:(j+1)*ks]
            psf = torch.flip(psf, [1, 2]).unsqueeze(1)  # shape [C, 1, ks, ks]
            
            h_low, w_low = int(i/grid*H), int(j/grid*W)
            h_high, w_high = int((i+1)/grid*H), int((j+1)/grid*W)
            
            # Consider overlap to avoid boundary artifacts
            img_pad_patch = img_pad[:, :, h_low:h_high+2*pad, w_low:w_high+2*pad]
            render_patch = nnF.conv2d(img_pad_patch, psf, groups=img.shape[1], padding='valid', bias=None)
            render_img[:, :, h_low:h_high, w_low:w_high] = render_patch

    return render_img


def local_psf_render(input, psf, kernel_size=11,val=False):
    """ Render an image with local PSF. Use the different PSF kernel for different pixels.
    
        Blurs image with dynamic Gaussian blur.

    Args:
        input (Tensor): The image to be blurred (N, C, H, W).
        psf (Tensor): Per pixel local PSFs (1, H, W, ks, ks)
        kernel_size (int): Size of the PSFs. Defaults to 11.

    Returns:
        output (Tensor): Rendered image (N, C, H, W)
    """
    if len(input.shape) < 4:
        input = input.unsqueeze(0)

    orig_dtype = input.dtype
    input = input.half()
    psf = psf.half()

    b,c,h,w = input.shape
    pad = int((kernel_size-1)/2)

    # 1. pad the input with replicated values
    inp_pad = torch.nn.functional.pad(input, pad=(pad,pad,pad,pad), mode='replicate')
    # 2. Create a Tensor of varying Gaussian Kernel
    kernels = psf.reshape(-1, 2, kernel_size, kernel_size)
    kernels_flip = torch.flip(kernels, [-2, -1])
    kernels_rgb = torch.stack(c*[kernels_flip], 2)
    # 3. Unfold input
    inp_unf = torch.nn.functional.unfold(inp_pad, (kernel_size,kernel_size))   
    # 4. Multiply kernel with unfolded
    x1 = inp_unf.view(b,c,-1,h*w)
    xl = kernels_rgb[:,0,...].view(b, h*w, c, -1).permute(0, 2, 3, 1)
    yl = (x1*xl).sum(2)
    rl = torch.nn.functional.fold(yl,(h,w),(1,1))
    xr = kernels_rgb[:,1,...].view(b, h*w, c, -1).permute(0, 2, 3, 1)
    yr = (x1*xr).sum(2)
    rr = torch.nn.functional.fold(yr,(h,w),(1,1))

    rl = rl.to(orig_dtype)
    rr = rr.to(orig_dtype)
    return rl,rr

def local_psf_render_fast(input, psf, kernel_size=11, val=False):
    if len(input.shape) < 4:
        input = input.unsqueeze(0)

    orig_dtype = input.dtype
    input = input.half()
    psf = psf.half()

    b, c, h, w = input.shape
    pad = (kernel_size - 1) // 2

    # 1. Pad the input
    inp_pad = torch.nn.functional.pad(input, pad=(pad, pad, pad, pad), mode='replicate')
    
    # 2. Reshape and flip kernels
    kernels = psf.reshape(b, h, w, 2, kernel_size, kernel_size)  # [B, H, W, 2, ks, ks]
    kernels_flip = torch.flip(kernels, [-2, -1])                  # Flip spatial dims
    kernels_flip = kernels_flip.reshape(b, h*w, 2, -1)            # [B, H*W, 2, ks*ks]

    # 3. Unfold input
    inp_unf = torch.nn.functional.unfold(inp_pad, (kernel_size, kernel_size))  # [B, C*ks*ks, H*W]
    inp_unf = inp_unf.view(b, c, -1, h*w)                        # [B, C, ks*ks, H*W]
    torch.cuda.empty_cache()
    
    # 4. Compute left and right blur
    # Left kernels: [B, H*W, ks*ks] -> [B, C, ks*ks, H*W] (broadcasted)
    yl = (inp_unf * kernels_flip[:, :, 0, :].permute(0, 2, 1).unsqueeze(1)).sum(2)
    rl = torch.nn.functional.fold(yl, (h, w), (1, 1))
    
    # Right kernels
    yr = (inp_unf * kernels_flip[:, :, 1, :].permute(0, 2, 1).unsqueeze(1)).sum(2)
    rr = torch.nn.functional.fold(yr, (h, w), (1, 1))

    rl = rl.to(orig_dtype)
    rr = rr.to(orig_dtype)
    return rl, rr

def local_dp_psf_render(input, dp_psf, kernel_size=21):
    """ Render DP image with local DP PSF. 
    Use the different DP PSF kernel for different pixels.
    Args:
        input (Tensor): The image to be blurred (N, C, H, W).
        psf (Tensor): Per pixel local DP PSFs (1, H, W, 2, ks, ks)
        kernel_size (int): Size of the DP PSFs. Defaults to 21.
    Returns:
        Output (Tensor): Rendered DP image (N, 2*C, H, W)
    """
    
    b,c,h,w = input.shape
    pad = int((kernel_size-1)/2)

    # 1. pad the input with replicated values
    inp_pad = torch.nn.functional.pad(input, pad=(pad,pad,pad,pad), mode='replicate')
    # 2. Create a Tensor of varying DP PSF
    kernels = dp_psf.reshape(-1, 2, kernel_size, kernel_size)
    kernels_flip = torch.flip(kernels, [-2, -1])
    kernels_rgb = torch.stack(c*[kernels_flip], 2)
    # 3. Unfold input
    inp_unf = torch.nn.functional.unfold(inp_pad, (kernel_size,kernel_size))   
    # 4. Multiply kernel with unfolded
    x1 = inp_unf.view(b,c,-1,h*w)
    x2_l = kernels_rgb[:,0,...].view(b, h*w, c, -1).permute(0, 2, 3, 1)
    x2_r = kernels_rgb[:,1,...].view(b, h*w, c, -1).permute(0, 2, 3, 1)
    y_l = (x1*x2_l).sum(2)
    y_r = (x1*x2_r).sum(2)
    render_l = torch.nn.functional.fold(y_l,(h,w),(1,1))
    render_r = torch.nn.functional.fold(y_r,(h,w),(1,1))

    return torch.cat([render_l,render_r], dim=1)


def local_psf_render_high_res(input, psf, patch_size=[320, 480], kernel_size=11):
    """ Patch-based rendering with local PSF. Use the different PSF kernel for different pixels.
    """
    B, C, H, W = input.shape
    img_render = torch.zeros_like(input)
    for pi in range(int(np.ceil(H/patch_size[0]))):    # int function here is not accurate
        for pj in range(int(np.ceil(W/patch_size[1]))):
            low_i = pi * patch_size[0]
            up_i = min((pi+1)*patch_size[0], H)
            low_j = pj * patch_size[1]
            up_j =  min((pj+1)*patch_size[1], W)

            img_patch = input[:, :, low_i:up_i, low_j:up_j]
            psf_patch = psf[:, low_i:up_i, low_j:up_j, :, :]

            img_render[:, :, low_i:up_i, low_j:up_j] = local_psf_render(img_patch, psf_patch, kernel_size=kernel_size)
    
    return img_render
