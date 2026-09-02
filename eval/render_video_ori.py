import os
import sys
import math
import pickle
import argparse
import time
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import csv
from numba import cuda

sys.path.append(os.getcwd())
from utils import *
from utils.logger import create_logger
from data_loaders.grab.utils.config import Config
from data_loaders.grab.utils.dataset_grab_action_transition import DatasetGrab
from data_loaders.grab.utils.dataset_ntu_act_transition import DatasetNTU
from data_loaders.grab.utils.dataset_humanact12_act_transition import DatasetACT12
from data_loaders.grab.utils.dataset_babel_action_transition import DatasetBabel
from eval.motion_pred import *
from utils.fid import calculate_frechet_distance
from utils.dtw import batch_dtw_torch, batch_dtw_torch_parallel, accelerated_dtw, batch_dtw_cpu_parallel
from utils import eval_util
from utils import data_utils
from utils.vis_util import render_videos_new
from utils.parser_util import generate_args
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from data_loaders.get_data import get_dataset_loader
from diffusion.resample import create_named_schedule_sampler
from tqdm import tqdm


def val(epoch):
    seq_len = []
    with torch.no_grad():
        for i, act in enumerate(dataset.act_name):
            st = time.time()
            
            #for traj_np, label, fn_gt, fn_mask_gt in generator:
            for motion, cond in data: 
                #print("cond",cond['y'].keys())
                #print("motion",motion.shape)
                Y = motion
                label = cond['y']['action']
                fn_gt = cond['y']['fn']
                fn_mask_gt = cond['y']['fn_mask']
                X = cond['y']['samp_his']
                # 拼接历史和未来动作序列 (traj_np)
                traj_np = torch.cat([X, Y], dim=-1)  # Y 是 ground-truth future
                bs, njoints, nfeats, seq_len = traj_np.shape
                traj_np = traj_np.permute(0, 3, 1, 2).reshape(bs, seq_len, njoints * nfeats)
                traj_tmp = traj_np.permute(1, 0, 2).contiguous().cpu().numpy()  # (seq_len, bs, dim)

                # 获取每个 sample 的真实结束帧
                seq_gt = np.where(fn_gt == 1)[1]  # 长度为 batch_size
                betas = np.zeros(10)

                n_action = cfg_classifier.vae_specs['n_action']
                for ii in range(args.batch_size):
                    for jj in range(n_action):
                        sequence = {
                            'poses': traj_tmp[:, ii],  # 使用真实的 GT + past 动作序列拼接
                            'betas': betas
                        }
                        key = f'{act}_{ii}_{dataset.act_name[jj]}_concat'
                        render_videos_new(sequence, device, os.path.join(cfg.result_dir, args.mode), key,
                                        w_golbalrot=True, smpl_model=smpl_model)


if __name__ == '__main__':
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')

    #parser = argparse.ArgumentParser()
    #parser.add_argument('--cfg', default='babel_rnn')
    

    #arg = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.data_type == 'float32':
        dtype = torch.float32
    else:
        dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
        cuda.select_device(args.gpu_index)
    cfg = Config(args.cfg, test=args.test)
    cfg_classifier = Config(args.cfg_classifier, test=args.test)
    # tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
    logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

    """parameter"""
    mode = args.mode
    nz = cfg.nz
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    if 't_pre_extra' in cfg.vae_specs:
        args.t_pre_extra = cfg.vae_specs['t_pre_extra']

    """data"""
    if cfg.dataset == 'grab':
        dataset_cls = DatasetGrab
        smpl_model = 'smplx'
    elif cfg.dataset == 'ntu':
        dataset_cls = DatasetNTU
        smpl_model = 'smpl'
    elif cfg.dataset == 'humanact12':
        dataset_cls = DatasetACT12
        smpl_model = 'smpl'
    elif cfg.dataset == 'babel':
        dataset_cls = DatasetBabel
        smpl_model = 'smplh'

    # for act in cfg.vae_specs['actions']:
    dataset = dataset_cls(args.mode, t_his, t_pred, actions='all', use_vel=cfg.use_vel,
                          acts=cfg.vae_specs['actions'] if 'actions' in cfg.vae_specs else None,
                          max_len=cfg.vae_specs['max_len'] if 'max_len' in cfg.vae_specs else None,
                          min_len=cfg.vae_specs['min_len'] if 'min_len' in cfg.vae_specs else None,
                          is_6d=cfg.vae_specs['is_6d'] if 'is_6d' in cfg.vae_specs else False,
                          data_file=cfg.vae_specs['data_file'] if 'data_file' in cfg.vae_specs else None)
    #print("dataset name",args.dataset)
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=60,hml_mode='video',cfg=cfg)
    """my model"""
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(device)
    model.eval()  # disable random masking


    """model"""
    """
    if cfg.dataset == 'babel':
        dataset.traj_dim = 60
        model = get_action_vae_model(cfg, 60, max_len=dataset.max_len - cfg.t_his + cfg.vae_specs['t_pre_extra'])
    else:
        model = get_action_vae_model(cfg, dataset.traj_dim, max_len=dataset.max_len - cfg.t_his + cfg.vae_specs['t_pre_extra'])
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    if args.iter > 0:
        cp_path = cfg.vae_model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])
    model.to(device)
    model.eval()
    """
    

    """action classifier model"""
    model_classifier = get_action_classifier(cfg_classifier, dataset.traj_dim, max_len=dataset.max_len)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in list(model_classifier.parameters())) / 1000000.0))
    cp_path = cfg_classifier.vae_model_path % (100 if cfg.dataset == 'babel' else 500)
    print('loading model from checkpoint: %s' % cp_path)
    model_cp = pickle.load(open(cp_path, "rb"))
    model_classifier.load_state_dict(model_cp['model_dict'])
    model_classifier.to(device)
    model_classifier.eval()

    val(args.iter)