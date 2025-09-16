""" 
Implicate representation of PSF.

Input [x, y, z]. Output [ks, ks] PSF kernel.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
from datetime import datetime
from deeplens.psfnet import PSFNet
from deeplens.utils import set_logger, set_seed
import sys
sys.dont_write_bytecode = True

result_dir = f'./results/' + datetime.now().strftime("%m%d-%H%M%S") + '-psfnet'
os.makedirs(result_dir, exist_ok=True)
set_logger(result_dir)
set_seed(0)

if __name__ == "__main__":
    ks = 21 # 21forf4, 35forf1.8
    psfnet = PSFNet(filename='./lenses/rf50mm/lens_web.json', sensor_res=(512, 768), kernel_size=ks, device='cuda')

    d_sensor = psfnet.d_sensor
    infocus = -1000 + d_sensor
    psfnet.refocus(infocus)

    psfnet.write_lens_json(f'{result_dir}/lens.json')
    print(psfnet.d_sensor)

    near_depth = -500 +d_sensor
    psfnet.analysis(save_name=f'{result_dir}/{int(near_depth)}',depth=near_depth, ks=ks)
    far_depth = -20000+d_sensor
    psfnet.analysis(save_name=f'{result_dir}/{int(far_depth)}',depth=far_depth, ks=ks)

    psfnet.load_net('./ckpt/F4_PSFNet_mlp.pkl')
    psfnet.train_psfnet(iters=90000, bs=64, lr=1e-4, spp=20000, evaluate_every=1000, result_dir=result_dir)
    
    psfnet.compare_psf()
    # psfnet.time_compare_psf()
    
    print('Finish PSF net fitting.')