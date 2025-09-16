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
                    distance = np.sqrt((sx - x_center)**2 + (sy - y_center)**2)   
                    if distance <= radius:  
                        intersection_areas[i, j] += subgrid_size**2  
    intersection_areas /=intersection_areas.sum()
    return intersection_areas  

def ker_disk(kersig, kersize):  
    circ = calculate_intersection_area(kersize,kersig)
    refcirc = np.zeros((kersize, kersize))  
    radius = np.abs(kersig)  
    
    # Center the circ on the refcirc.
    center_row = (kersize - circ.shape[0]) // 2  
    center_col = (kersize - circ.shape[1]) // 2  
    refcirc[center_row:center_row+circ.shape[0], center_col:center_col+circ.shape[1]] = circ  

    # Similar to MATLAB linspace
    dist_array = np.arange(0, 2*radius+2)  
    diskker = np.zeros_like(refcirc)  
      
    # Similar to the effect of imtranslate, translate the refcirc and accumulate the results.
    for i in dist_array:  
        shift_row = int(np.sign(kersig) * i)  
        shifted = shift(refcirc, [0,shift_row], cval=0, mode='constant')  
        add_ = refcirc*shifted
        diskker += add_  
    kerout = 0.5 * diskker / diskker.sum()  
    psf_l, psr_r = kerout, np.flip(kerout)
    return psf_l, psr_r  
  

if __name__ == "__main__":
    kersig = 0.2  # Example sigma value  
    kersize = 21  # Example kernel size
    psf_l, psr_r = ker_disk(kersig, kersize)  
    # output = output/output.max()
    plt.imshow(psf_l,cmap="bone")
    plt.show()
