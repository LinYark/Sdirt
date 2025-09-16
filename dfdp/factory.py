from deeplens.psfnet import *
from dfdp.dataset import *

def get_lens(args):
    ks = args['ks']
    sensor_res = args['res']
    device = args['device']

    train_lens_name = args['train']['lens']
    if train_lens_name == 'thinlens':
        foc_len = args['train']['foc_len']
        fnum = args['train']['fnum']
        sensor_size = [float(i) for i in args['train']['sensor_size']]
        train_lens = ThinLens(foc_len=foc_len, fnum=fnum, kernel_size=ks, sensor_size=sensor_size, sensor_res=sensor_res)
        train_lens = train_lens.to(device)
    else:
        train_lens = PSFNet(filename=train_lens_name, sensor_res=sensor_res, kernel_size=ks,device=device)
        train_lens.load_net(args['train']['psfnet_path'])

    test_lens_name = args['test']['lens']
    if test_lens_name == 'thinlens':
        foc_len = args['test']['foc_len']
        fnum = args['test']['fnum']
        sensor_size = [float(i) for i in args['test']['sensor_size']]
        test_lens = ThinLens(foc_len=foc_len, fnum=fnum, kernel_size=ks, sensor_size=sensor_size, sensor_res=sensor_res)
        test_lens = test_lens.to(device)
    else:
        test_lens = PSFNet(filename=test_lens_name, sensor_res=sensor_res, kernel_size=ks, device=device)
        test_lens.load_net(args['test']['psfnet_path'])
        
    train_lens.eval() 
    test_lens.eval()
    return train_lens, test_lens

def get_dataset(args):
    train_dataset_name = args['train']['dataset']
    if train_dataset_name == 'FlyingThings3D':
        train_set = FlyingThings3D(args['FlyingThings3D_train'], resize=args['res'])
    elif train_dataset_name == 'NYUdata':
        train_set = NYUData(args['NYUdata_train'], resize=args['res'])
    else:
        raise NotImplementedError

    test_dataset_name = args['test']['dataset']
    if test_dataset_name == 'Middlebury2014':
        test_set = Middlebury(args['Middlebury2014_val'], resize=args['res'], train=False)
    elif test_dataset_name == 'Middlebury2021':
        test_set = Middlebury(args['Middlebury2021_val'], resize=args['res'], train=False)
    elif test_dataset_name == 'Middlebury_FS':
        test_set = Middlebury_FS(args['Middlebury_FS'], resize=args['res'], train=False)
    elif test_dataset_name == 'FlyingThings3D':
        test_set = FlyingThings3D(args['FlyingThings3D_test'], resize=args['res'], train=False)
    elif test_dataset_name == 'NYUdata':
        test_set = NYUData(args['NYUdata_test'], resize=args['res'], train=False)
    else:
        raise NotImplementedError

    fly_train_set = FlyingThings3D(args['FlyingThings3D_train'], resize=args['res'])
    nyu_fs_train_set = train_set + fly_train_set + fly_train_set # nyu+flythings
    nyu_train_set = train_set + train_set # nyu
    return nyu_fs_train_set, nyu_train_set, test_set

def get_depth_test_set(args):
    dataset_dir = args['real_box_test']
    box_set = Canon_Depth_Set(dataset_dir, resize=args['res'])
    dataset_dir = args['real_flat_test']
    flat2depth_set = Canon_Flat2Depth_Set(dataset_dir, resize=args['res'])
    dataset_dir = args['real_casual_test']
    casual_set = Canon_Casual_Set(dataset_dir, resize=args['res'])
    return box_set, flat2depth_set, casual_set

def get_flat_test_set(args):
    dataset_dir = args['real_flat_test']
    test_set = Canon_Flat_Set(dataset_dir, resize=args['res'])
    return test_set

def get_depth_sample_set(args):
    dataset_dir = args['real_box_sample']
    box_set = Canon_Depth_Set(dataset_dir, resize=args['res'])
    dataset_dir = args['real_flat_sample']
    flat2depth_set = Canon_Flat2Depth_Set(dataset_dir, resize=args['res'])
    dataset_dir = args['real_casual_sample']
    casual_set = Canon_Casual_Set(dataset_dir, resize=args['res'])
    return box_set, flat2depth_set, casual_set

def get_flat_sample_set(args):
    dataset_dir = args['real_flat_sample']
    test_set = Canon_Flat_Set(dataset_dir, resize=args['res'])
    return test_set