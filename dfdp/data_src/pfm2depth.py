"""
    Modify from https://github.com/singer-yang/Middlebury-Depth-Map?tab=readme-ov-file
    linyark 2024/10/08
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv
import numpy as np
import re
import png
from glob import glob
import matplotlib.pyplot as plt

def process_middlebury2021():
    scenes = glob('./Middlebury2021/*')
    for scene in scenes:
        process_pfm(scene)


def process_middlebury2014():
    scenes = glob('./Middlebury2014/*')
    for scene in scenes:
        process_pfm(scene)

def show():
    aif_img = cv.cvtColor(cv.imread(f'./Middlebury2014/Adirondack-perfect/im0.png'), cv.COLOR_BGR2RGB)
    depth = cv.imread(f'./Middlebury2014/Adirondack-perfect/depth.png', -1) / 1000
    depth[depth<=0] = depth.max()
    plt.figure()
    plt.imshow(aif_img)
    plt.figure()
    plt.imshow(depth,cmap='Spectral')
    plt.show()

def process_pfm(scene):
    """ Read Middlebury disparity map and calculate depth map.
    
        http://davis.lbl.gov/Manuals/NETPBM/doc/pfm.html
    """
    # ==> Read disparity map from *.pfm file
    disp_path = f'{scene}/disp0.pfm'
    disp = cv.imread(disp_path, cv.IMREAD_UNCHANGED)
    with open(disp_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channels = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(pfm_file.readline().decode().rstrip())    # read disparity scale factor
        if scale < 0:
            endian = '<' # littel endian
            scale = -scale
        else:
            endian = '>' # big endian

    disp = disp * scale

    # # show as paper
    # dis1 = disp*1
    # dis1[dis1==np.inf]=0
    # plt.figure()
    # plt.imshow(dis1,cmap='jet')
    # plt.show()

    # ==> Read calibration file
    calib = f'{scene}/calib.txt'
    with open(calib, 'r') as f:
        lines = f.readlines()

    f = float(re.findall("\d+\.\d+", lines[0])[0])
    try:
        doffs = float(re.findall("\d+\.\d+", lines[2])[0])
    except:
        doffs = float(re.findall("\d+", lines[2])[0])
    baseline = float(re.findall("\d+\.\d+", lines[3])[0])
    
    # ==> Calculate depth map
    depth = baseline * f / (disp + doffs) # depth in [mm]

    # ==> Write depth map into 16-bit .png
    depth_path = f'{scene}/depth.png'
    depth_uint16 = np.round(depth).astype(np.uint16)
    with open(depth_path, 'wb') as f:
        writer = png.Writer(width=width, height=height, bitdepth=16, greyscale=True)
        writer.write(f, np.reshape(depth_uint16, (-1, width)))

    return depth

def t_pfm():
    """ Read Middlebury disparity map and calculate depth map.
    
        http://davis.lbl.gov/Manuals/NETPBM/doc/pfm.html
    """
    # ==> Read disparity map from *.pfm file
    disp_path = f'./disp.exr'
    DEPTH_FACTOR = 10
    disp = cv.imread(disp_path, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH) / DEPTH_FACTOR

    # show as paper
    dis1 = disp*1
    dis1[dis1==np.inf]=0
    plt.figure()
    plt.imshow(dis1,cmap='jet')
    plt.show()


if __name__=='__main__':
    # process_middlebury2014()
    t_pfm()
    # show()