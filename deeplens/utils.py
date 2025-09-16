import os
import random 
import numpy as np
import torch
import lpips
import logging
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# ==================================
# Image batch quality evaluation
# ==================================

def batch_PSNR(img_clean, img):
    """ Compute PSNR for image batch.
    """
    Img = img.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    Img_clean = img_clean.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Img_clean[i,:,:,:], Img[i,:,:,:])
    return round(PSNR/Img.shape[0], 4)


def batch_SSIM(img, img_clean, multichannel=True):
    """ Compute SSIM for image batch.
    """
    Img = img.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    Img_clean = img_clean.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += compare_ssim(Img_clean[i,...], Img[i,...], channel_axis=0)
    
    return round(SSIM/Img.shape[0], 4)


def batch_LPIPS(img, img_clean):
    """ Compute LPIPS loss.
    """
    device = img.device
    loss_fn = lpips.LPIPS(net='vgg', spatial=True)
    loss_fn.to(device)
    dist = loss_fn.forward(img, img_clean)
    return dist.mean().item()


def calculate_psnr(img1,img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
def calculate_ssim(img1,img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


# ==================================
# Image batch normalization
# ==================================

def normalize_ImageNet_stats(batch):
    """ Normalize dataset by ImageNet(real scene images) distribution. 
    """
    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    
    batch_out = (batch - mean) / std
    return batch_out


def de_normalize(batch):
    """ Convert normalized images to original images to compute PSNR.
    """
    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    
    batch_out = batch * std + mean
    return batch_out
     
def gpu_init(gpu=0):
    """Initialize device and data type.

    Returns:
        device: which device to use.
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print("Using: {}".format(device))
    torch.set_default_tensor_type('torch.FloatTensor')
    return device


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def set_logger(dir='./'):
    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')

    fhlr = logging.FileHandler(f"{dir}/output.log")
    fhlr.setFormatter(formatter)
    fhlr.setLevel('INFO')

    logger.addHandler(chlr)
    logger.addHandler(fhlr)

def print_memory():
    """ Print CUDA memory consumption, already replaced by gpustat.
    """
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f'reserved memory: ~{r * 1e-9}GB, free memory: ~{f * 1e-9}GB.')