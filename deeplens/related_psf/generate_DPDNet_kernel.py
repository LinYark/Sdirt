import numpy as np  
from scipy.ndimage import shift   
import cv2
import matplotlib.pyplot as plt

def calculate_intersection_area(ks, radius,grid_size=1, subgrid_size=0.25 ):  
    grid_x = np.linspace(0, ks-1, ks) 
    grid_y = np.linspace(0, ks-1, ks) 
    x_center, y_center = ks/2.0, ks/2.0

    nx, ny = len(grid_x), len(grid_y)  
    intersection_areas = np.zeros((nx, ny))   
    for i in range(nx):  
        for j in range(ny):   
            x_min, x_max = grid_x[i], grid_x[i] + grid_size  
            y_min, y_max = grid_y[j], grid_y[j] + grid_size  
   
            for sx in np.arange(x_min, x_max, subgrid_size):  
                sx = sx+subgrid_size/2
                for sy in np.arange(y_min, y_max, subgrid_size):  
                    sy = sy+subgrid_size/2
                    if sx>=x_center-radius and sx<=x_center+radius and \
                        sy>=y_center-radius and sy<=y_center:
                        intersection_areas[i, j] += subgrid_size**2  
    intersection_areas /=intersection_areas.sum()
    return intersection_areas  

def ker_rect(kersig, kersize):  
    kerout = calculate_intersection_area(kersize,kersig) 
    psf_l, psr_r = kerout, np.flip(kerout)
    return psf_l, psr_r  
  

if __name__ == "__main__":
    kersig = 3.7  # Example sigma value  
    kersize = 21  # Example kernel size
    psf_l, psr_r = ker_rect(kersig, kersize)  
    plt.imshow(psf_l,cmap="bone")
    plt.show()
