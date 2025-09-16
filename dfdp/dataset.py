import torch
from torch.utils.data import Dataset
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv
import numpy as np
from glob import glob
from torchvision import transforms
from skimage.morphology import disk, closing
import random
from scipy.ndimage.interpolation import rotate
import torch.nn.functional as F

# ================================
# Dataset
# ================================

class NYUData(Dataset):
    def __init__(self, rgb_path, resize=None, train=True):
        super(NYUData, self).__init__()

        self.rgb_path = rgb_path
        self.depth_path = rgb_path
        self.scenes = glob(f'{rgb_path}/*')
        self.resize = resize
        self.train = train
        
        self.imgs = []
        self.depths = []
        for scene in self.scenes:
            imgs = sorted(glob(f'{scene}/*.jpg'))
            depths = sorted(glob(f'{scene}/*.png'))
            self.imgs += imgs
            self.depths += depths

        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC, antialias=True)
        ])
        self.transform_d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.NEAREST, antialias=False)
        ])

        self.scale = 25.5
        self.crop = 20

    def __len__(self):
        if self.train is True:
            return 2000
        else:
            return 50
        
    def __getitem__(self, idx):
        if self.train==True:
            idx = np.random.randint(0, high=len(self.imgs))
        try:
            aif_img = cv.cvtColor(cv.imread(self.imgs[idx]), cv.COLOR_BGR2RGB) / 255.
            depth = cv.imread(self.depths[idx], -1) / self.scale  #* 1e3 # convert to [mm]
            h,w,c = aif_img.shape
            aif_img = aif_img[self.crop:(h-self.crop),self.crop:(w-self.crop),:]
            depth = depth[self.crop:(h-self.crop),self.crop:(w-self.crop)]
            assert (depth[depth>0].any()==True)
        except:
            print(f"fail file {self.depths[idx]}")
            return self.__getitem__(idx+1)

        if self.train:
            aif_img, depth = AutoAgument(aif_img, depth)
        depth = depth_preprocess(depth)
        
        aif_img = self.transform_rgb(aif_img.astype('float32'))
        depth = self.transform_d(depth.astype('float32'))
        return [aif_img, depth]

class FlyingThings3D(Dataset):
    def __init__(self, dataset_dir, resize=None, train=True, fs_num=0):
        super(FlyingThings3D, self).__init__()

        self.dataset_dir = dataset_dir
        self.scenes = [scene.split('/')[-1] for scene in glob(f'{dataset_dir}/*')]
        self.resize = resize
        self.fs_num = fs_num
        self.train = train

        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC, antialias=True)
        ])
        self.transform_d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.NEAREST, antialias=False)
        ])

    def __len__(self):
        # return 400
        lenth = len(self.scenes)
        if self.train is False:
            lenth = 50
        return lenth

    def __getitem__(self, index):
        scene = self.scenes[index]
        dataset_dir = self.dataset_dir
        DEPTH_FACTOR = 20
        resize = [self.resize[1], self.resize[0]]

        depth = cv.resize(cv.imread(f'{dataset_dir}/{scene}/disp.exr', cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH) / DEPTH_FACTOR, resize) #m
        
        if self.fs_num > 0:
            focused_imgs = []
            focal_dists = []
            full_focal_stack = sorted(glob(f'{dataset_dir}/{scene}/*.png'))[:-1]
            selected_imgs = random.sample(full_focal_stack, self.fs_num)
            for img_name in selected_imgs:
                focal_dists.append(float(img_name.split('/')[-1][:-4]) / DEPTH_FACTOR)
                focused_img = cv.resize(cv.imread(img_name).astype(np.float32)/255., resize)
                focused_imgs.append(focused_img)
            
            focal_stack = np.stack(focused_imgs, axis=-1)
            
            if self.train:
                focal_stack, depth = AutoAgument(focal_stack, depth)
                
            focal_stack = np.transpose(focal_stack, (3, 2, 0, 1))   # shape of (S, C, H, W)
            focal_stack = torch.from_numpy(focal_stack.astype('float32'))
            depth = torch.from_numpy(depth.astype('float32')).unsqueeze(0)
            focal_dists = torch.from_numpy(np.stack(focal_dists, axis=-1))    
            return [focal_stack, depth, focal_dists]

        else:
            aif_img = cv.cvtColor(cv.imread(f'{dataset_dir}/{scene}/AiF.png'), cv.COLOR_BGR2RGB) / 255.
            
            if self.train:
                aif_img, depth = AutoAgument(aif_img, depth)
            depth = depth_preprocess(depth)

            aif_img = self.transform_rgb(aif_img.astype('float32'))
            depth = self.transform_d(depth.astype('float32'))
            return [aif_img, depth]

class Middlebury_FS(Dataset):
    def __init__(self, dataset_dir, resize=None, train=False, fs_num=0):
        super(Middlebury_FS).__init__()

        self.dataset_dir = dataset_dir
        self.scenes = [scene.split('/')[-1] for scene in glob(f'{dataset_dir}/*')]
        self.resize = resize
        self.fs_num = fs_num
        self.train = train

        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC, antialias=True)
        ])
        self.transform_d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.NEAREST, antialias=False)
        ])


    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        scene = self.scenes[index]
        DEPTH_FACTOR = 10   # convert disparity to depth
        resize = [self.resize[1], self.resize[0]]

        depth = cv.resize(cv.imread(f'{dataset_dir}/{scene}/disp.exr', cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH) / DEPTH_FACTOR, resize)

        if self.fs_num > 0:
            raise Exception('Untested.')
            focused_imgs = []
            focal_dists = []
            full_focal_stack = sorted(glob(f'{dataset_dir}/{scene}/*.png'))[:-1]
            for _ in range(self.fs_num):
                focused_img = random.choice(full_focal_stack)
                focal_dists.append(float(focused_img.split('/')[-1][:-4]) / DEPTH_FACTOR)
                focused_img = cv.resize(cv.imread(focused_img).astype(np.float32)/255., resize)
                focused_imgs.append(focused_img)
            
            focal_stack = np.stack(focused_imgs, axis=-1)
            
            if self.train:
                focal_stack, depth = AutoAgument(focal_stack, depth)
                
            focal_stack = np.transpose(focal_stack, (3, 2, 0, 1))
            focal_stack = torch.from_numpy(focal_stack.astype('float32'))
            depth = torch.from_numpy(depth.astype('float32')).unsqueeze(0)
            focal_dists = torch.from_numpy(np.stack(focal_dists, axis=-1))
            
            return [focal_stack, depth, focal_dists]

        else:
            aif_img = cv.cvtColor(cv.imread(f'{dataset_dir}/{scene}/AiF.png'), cv.COLOR_BGR2RGB) / 255.
            
            if self.train:
                aif_img, depth = AutoAgument(aif_img, depth)
            depth[depth<0] = 0
            aif_img = self.transform_rgb(aif_img.astype('float32'))
            depth = self.transform_d(depth.astype('float32'))

            return [aif_img, depth]


class Middlebury(Dataset):
    def __init__(self, dataset_dir, resize=None, train=False):
        super(Middlebury).__init__()

        self.dataset_dir = dataset_dir
        self.scenes = sorted([scene.split('/')[-1] for scene in glob(f'{dataset_dir}/*')])
        self.resize = resize
        self.train = train

        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC, antialias=True)
        ])
        self.transform_d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.NEAREST, antialias=False)
        ])

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        scene = self.scenes[index]
        resize = [self.resize[1], self.resize[0]]

        aif_img = cv.cvtColor(cv.imread(f'{dataset_dir}/{scene}/im0.png'), cv.COLOR_BGR2RGB) / 255.
        depth = cv.resize(cv.imread(f'{dataset_dir}/{scene}/depth.png', -1) / 1000, resize)

        aif_img = self.transform_rgb(aif_img.astype('float32'))
        depth = self.transform_d(depth.astype('float32'))

        return [aif_img, depth]


# ================================
# Data augmentation
# ================================
def AutoAgument(img, depth):
    """ Automatic data augmentation.

    Args:
        img: [H, W, 3] ndarray
        depth: [H, W] ndarray 
    """
    # Color jitter
    if np.random.rand() > 0.5:
        contrast = np.random.uniform(0.75, 1.25) 
        brightness = np.random.uniform(-0.25, 0.25)
        img = contrast * img + brightness
        img = np.clip(img, 0.0, 1.0)
    
    # Gamma
    if np.random.rand() > 0.5:
        gamma_down = np.random.uniform(0.5,1)
        gamma_up = np.random.uniform(1,2)
        gamma = gamma_up if np.random.rand()>0.5 else gamma_down
        img = img**gamma

    # Flip W
    if np.random.rand() > 0.5:
        img = np.flip(img, 1)
        depth = np.flip(depth, 1)

    # Flip H
    if np.random.rand() > 0.75:
        img = np.flip(img, 0)
        depth = np.flip(depth, 0)

    # # Rotate
    # if np.random.rand() > 0.75:
    #     degree = np.random.randint(0, 180)
    #     if len(img.shape) == 4:
    #         for i in range(img.shape[-1]):
    #             img[...,i] = rotate(img[..., i], degree, reshape=False)
    #     else:
    #         img = rotate(img, degree, reshape=False)
    #     depth = rotate(depth, degree, reshape=False)

    # Crop
    if np.random.rand()>0.5:
        limit = 20
        shift = np.random.randint(0,limit)
        h,w,c = img.shape
        img = img[shift:(h-(limit-shift)),shift:(w-(limit-shift)),:]
        depth = depth[shift:(h-(limit-shift)),shift:(w-(limit-shift))]

    # # depth_flat
    # if np.random.rand() > 0.5:
    #     target_depth = np.random.rand()*(d_max-d_min)+d_min
    #     scale = np.random.rand()*0.1 + 0.9
    #     depth[depth>0]  = depth[depth>0] +(target_depth-depth[depth>0])*scale
    
    # depth_shift
    if np.random.rand() > 0.5:
        times = np.random.uniform(0.25,1.25)
        depth = depth * times

    return img, depth

def depth_preprocess(depth):
    scale = 1.0
    depth = depth / scale
    depth_mark = depth*1.0
    depth = np.clip(depth, 0.25, 10)
    
    depth[depth_mark<=0] = 0
    return depth

class Canon_Depth_Set(Dataset):
    def __init__(self, dataset_dir, resize=None):
        super(Canon_Depth_Set, self).__init__()

        self.dataset_dir = dataset_dir
        scenes = glob(f'{dataset_dir}/*')
        self.scenes = sorted(scenes)
        # [scene.split('/')[-1] for scene in glob(f'{dataset_dir}/*')]
        self.resize = resize
        self.file_type = glob(f"{self.scenes[0]}/l.*")[0].split('.')[-1]
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC, antialias=True)
        ])
        self.transform_d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.NEAREST, antialias=False)
        ])


    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        scene = self.scenes[index]
        DEPTH_FACTOR = 10
        resize = [self.resize[1], self.resize[0]]

        if os.path.exists(f'{scene}/d.png'):
            depth = cv.resize(cv.imread(f'{scene}/d.png',0)/255.0 * DEPTH_FACTOR, resize)
        else:
            depth = np.ones(resize, dtype=np.float64)*2.5
        l_img = cv.cvtColor(cv.imread(f'{scene}/l.{self.file_type}'), cv.COLOR_BGR2RGB) / 255.
        r_img = cv.cvtColor(cv.imread(f'{scene}/r.{self.file_type}'), cv.COLOR_BGR2RGB) / 255.

        l_img = self.transform_rgb(l_img.astype('float32'))
        r_img = self.transform_rgb(r_img.astype('float32'))
        img = torch.cat((l_img,r_img),dim=0)

        depth[depth<0] = 0
        depth[depth>=10] = 0
        depth = self.transform_d(depth.astype('float32'))
        return [img, depth]
    

class Canon_Flat2Depth_Set(Dataset):
    def __init__(self, dataset_dir, resize=None, train=True):
        super(Canon_Flat2Depth_Set, self).__init__()
        self.dataset_dir = dataset_dir
        img_paths = glob(f'{dataset_dir}/**/f4/l.*',recursive=True)
        self.file_type = img_paths[0].split('.')[-1]
        img_paths = sorted(img_paths)
        self.dis_l, self.imgp_l = [], []

        from os.path import dirname,basename
        for img_path in img_paths:
            dis_str = basename(dirname(dirname(img_path)))
            if "inf" in dis_str:
                continue
            dis = float(dis_str)
            dis_m = dis/1000.0
            self.dis_l.append(dis_m)
            imgp = dirname(dirname(img_path))
            self.imgp_l.append(imgp)

        self.resize = resize
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC, antialias=True)
        ])
        self.transform_d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.NEAREST, antialias=False)
        ])


    def __len__(self):
        return len(self.imgp_l)

    def __getitem__(self, index):
        dis_m, imgp = self.dis_l[index], self.imgp_l[index]
        resize = [self.resize[1], self.resize[0]]
        f4_l_img = cv.cvtColor(cv.imread(f'{imgp}/f4/l.{self.file_type}'), cv.COLOR_BGR2RGB) / 255.
        f4_r_img = cv.cvtColor(cv.imread(f'{imgp}/f4/r.{self.file_type}'), cv.COLOR_BGR2RGB) / 255.

        f4_l_img = self.transform_rgb(f4_l_img.astype('float32'))
        f4_r_img = self.transform_rgb(f4_r_img.astype('float32'))
        f4_img = torch.cat((f4_l_img, f4_r_img), dim=0)

        depth = np.ones(resize)*dis_m
        depth = self.transform_d(depth.astype('float32'))
        return [f4_img, depth]

class Canon_Flat_Set(Dataset):
    def __init__(self, dataset_dir, resize=None, train=True):
        super(Canon_Flat_Set, self).__init__()
        inf = 100000
        self.dataset_dir = dataset_dir
        img_paths = glob(f'{dataset_dir}/**/f4/l.*',recursive=True)
        self.file_type = img_paths[0].split('.')[-1]
        img_paths = sorted(img_paths)
        self.dis_l, self.imgp_l = [], []

        from os.path import dirname,basename
        for img_path in img_paths:
            dis_str = basename(dirname(dirname(img_path)))
            dis = inf if "inf" in dis_str else float(dis_str)
            dis_m = dis/1000.0
            self.dis_l.append(dis_m)
            imgp = dirname(dirname(img_path))
            self.imgp_l.append(imgp)

        self.resize = resize
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC, antialias=True)
        ])
        self.transform_d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.NEAREST, antialias=False)
        ])


    def __len__(self):
        return len(self.imgp_l)

    def __getitem__(self, index):
        dis_m, imgp = self.dis_l[index], self.imgp_l[index]
        resize = [self.resize[1], self.resize[0]]
        f4_l_img = cv.cvtColor(cv.imread(f'{imgp}/f4/l.{self.file_type}'), cv.COLOR_BGR2RGB) / 255.
        f4_r_img = cv.cvtColor(cv.imread(f'{imgp}/f4/r.{self.file_type}'), cv.COLOR_BGR2RGB) / 255.
        f20_l_img = cv.cvtColor(cv.imread(f'{imgp}/f20/l.{self.file_type}'), cv.COLOR_BGR2RGB) / 255.
        f20_r_img = cv.cvtColor(cv.imread(f'{imgp}/f20/r.{self.file_type}'), cv.COLOR_BGR2RGB) / 255.


        f4_l_img = self.transform_rgb(f4_l_img.astype('float32'))
        f4_r_img = self.transform_rgb(f4_r_img.astype('float32'))
        f20_l_img = self.transform_rgb(f20_l_img.astype('float32'))
        f20_r_img = self.transform_rgb(f20_r_img.astype('float32'))

        f4_img = torch.cat((f4_l_img, f4_r_img), dim=0)
        f20_img = torch.cat((f20_l_img, f20_r_img), dim=0)

        depth = np.ones(resize)*dis_m
        depth = self.transform_d(depth.astype('float32'))
        return [f4_img, f20_img, depth]
    
class Canon_Casual_Set(Dataset):
    def __init__(self, dataset_dir, resize=None):
        super(Canon_Casual_Set, self).__init__()

        self.dataset_dir = dataset_dir
        scenes = glob(f'{dataset_dir}/*/*')
        self.scenes = sorted(scenes)
        self.resize = resize
        self.file_type = glob(f"{self.scenes[0]}/l.*")[0].split('.')[-1]
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC, antialias=True)
        ])
        self.transform_d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.NEAREST, antialias=False)
        ])


    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        scene = self.scenes[index]
        DEPTH_FACTOR = 10
        resize = [self.resize[1], self.resize[0]]

        if 'iphone' in scene:
            depth = cv.resize(cv.imread(f'{scene}/d.png',0)/255.0 * DEPTH_FACTOR, resize)
        if 'orbbec' in scene:
            depth = cv.resize(cv.imread(f'{scene}/d.png',cv.IMREAD_UNCHANGED)/1000.0, resize)
        l_img = cv.cvtColor(cv.imread(f'{scene}/l.{self.file_type}'), cv.COLOR_BGR2RGB) / 255.
        r_img = cv.cvtColor(cv.imread(f'{scene}/r.{self.file_type}'), cv.COLOR_BGR2RGB) / 255.

        l_img = self.transform_rgb(l_img.astype('float32'))
        r_img = self.transform_rgb(r_img.astype('float32'))
        img = torch.cat((l_img,r_img),dim=0)

        depth[depth<0] = 0
        depth[depth>=10] = 0
        depth = self.transform_d(depth.astype('float32'))
        return [img, depth]
    
if __name__ == "__main__":
    dataset_dir = "./canon_calib/dataset/Release_flatset"
    a = Canon_Flat_Set(dataset_dir,resize=[512, 768])
    x = next(iter(a))
    a = 1