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
import copy
from numba import cuda
import kornia.geometry as torchgeometry
import torch
import utils.rotation_conversions as geometry
import numpy as np

sys.path.append(os.getcwd())
from utils import dist_util
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
from model.cfg_sampler import ClassifierFreeSampleModel


def get_stop_sign(Y_r,args):
    # get stop sign
    if args.stop_fn > 0:
        fn_tmp = Y_r.shape[0]
        tmp1 = np.arange(fn_tmp)[:, None]
        tmp2 = np.arange(args.stop_fn)[None, :]
        idxs = tmp1 + tmp2
        idxs[idxs > fn_tmp - 1] = fn_tmp - 1
        yr_tmp = Y_r[idxs]
        yr_mean = yr_tmp.mean(dim=1, keepdim=True)
        dr = torch.mean(torch.norm(yr_tmp - yr_mean, dim=-1), dim=1)
    else:
        dr = torch.norm(Y_r[:-1] - Y_r[1:], dim=2)
        dr = torch.cat([dr[:1, :], dr], dim=0)
    threshold = args.threshold
    tmp = dr < threshold
    idx = torch.arange(tmp.shape[0], 0, -1, device=device)[:, None]
    tmp2 = tmp * idx
    tmp2[:dataset.min_len - 1] = 0
    tmp2[-1, :] = 1
    fn = tmp2 == tmp2.max(dim=0, keepdim=True)[0]
    fn = fn.float()
    return fn
    
def rot6d_to_axisangle(rot6d):
    """
    Args:
        rot6d: Tensor of shape [B, J, 6] or [*, 6]

    Returns:
        axis_angle: Tensor of shape [B, J, 3] or [*, 3]
    """
    # rot6d → rotation matrix → axis-angle
    rotmat = geometry.rotation_6d_to_matrix(rot6d)             # [*, 3, 3]
    axis_angle = geometry.matrix_to_axis_angle(rotmat)         # [*, 3]
    return axis_angle



def extract_joint_traj(sequence_tensor, joint_idx=7, dim=1):
    """
    提取指定关节在指定维度的轨迹。
    Args:
        sequence_tensor: [T, B, J*3]
        joint_idx: 关节索引（默认左脚踝是7）
        dim: 提取的维度（x=0, y=1, z=2）
    Returns:
        traj: [T]，表示该关节的轨迹
    """
    joint_dim = 3
    traj = sequence_tensor[:, 0, joint_idx * joint_dim + dim]
    return traj.cpu().numpy()

def plot_motion_trajectories(history, pred_trans,
                              stop_frames=None, joint_idx=7, dim=1, save_path=None):
    """
    仅绘制历史轨迹与你自己的模型预测轨迹
    """
    plt.figure(figsize=(10, 5))

    # 提取轨迹
    hist_y = extract_joint_traj(history, joint_idx, dim)
    trans_y = extract_joint_traj(pred_trans, joint_idx, dim)

    x_len = len(hist_y)
    frames = np.arange(x_len + len(trans_y))

    # 历史与预测轨迹
    plt.plot(np.arange(x_len), hist_y, label='History', color='blue', linewidth=2)
    plt.plot(np.arange(x_len, x_len + len(trans_y)), trans_y, label='Ours (Transition)', color='green', linewidth=2)

    # 绘制停止帧
    if stop_frames is not None:
        for f in stop_frames:
            if f >= x_len and (f - x_len) < len(trans_y):
                plt.plot(f, trans_y[f - x_len], 'b*', markersize=10)

    plt.xlabel('Frame')
    plt.ylabel(f'Joint-{joint_idx} Position (axis {dim})')
    plt.title('Joint Trajectory Comparison')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


# 在可视化上，我应该对于一个动作，给定当前动作的历史序列。同时应该获取到dataset.act_name+1次模型输出，同时标签给的也是与dataset.act_name+1对应的label
# 而不一直是当前动作的label。
def val(epoch):
    seq_len = []
    with torch.no_grad():
        for i, act in enumerate(dataset.act_name):
            st = time.time()
            generator = dataset.sampling_generator(num_samples=args.num_samp, batch_size=args.bs,t_pre_extra=args.t_pre_extra,
                                                   act=act)

            #for traj_np, label, fn_gt, fn_mask_gt in generator:
            for motion, label, fn_gt, fn_mask_gt ,cond in generator:
            #for motion, cond in data: 
                #print("cond",cond['y'].keys())
                #print("motion",motion.shape)
                Y = motion
                #new_shape = Y.shape[:-1] + (max_len,)
                #Z = torch.zeros(new_shape, dtype=Y.dtype, device=Y.device)
                label = cond['y']['action']
                fn_gt = cond['y']['fn']
                fn_mask_gt = cond['y']['fn_mask']
                X = cond['y']['samp_his']
                seq_gt = np.where(fn_gt == 1)[1]
                #获取到所有动作的标签
                #label = torch.eye(cfg.vae_specs['n_action'], device=device, dtype=dtype)
                #label = label[None, :, None, :].repeat([bs, 1, args.nk, 1]).reshape([-1, cfg.vae_specs['n_action']])

                traj_np = torch.cat([X,motion],dim=-1)
                traj_np = traj_np.permute(0, 3, 1, 2).contiguous()  # [B, J, 6, T] → [B, T, J, 6]
                traj_np = rot6d_to_axisangle(traj_np)  # [B, T, J, 6]→[B, T, J, 3]
                B,T,J,F = traj_np.shape
                traj_np = traj_np.permute(1, 0, 2, 3).contiguous().view(T, B, J * F)  # # [B, T, J, 3]→[T, B, J*3]

                traj_tmp = tensor(traj_np, device=device, dtype=dtype).contiguous()
                seq_n, bs, dim = traj_tmp.shape

                # get history motion sequence
                X = X.permute(0, 3, 1, 2).contiguous()  # [B, J, 6, T] → [B, T, J, 6]
                X = rot6d_to_axisangle(X)  # [B, T, J, 6]→[B, T, J, 3]
                B,T_his,J,F = X.shape
                X = X.permute(1, 0, 2, 3).contiguous().view(T_his, B, J * F)  # # [B, T, J, 3]→[T, B, J*3]
                X_list = []


                # diffusion  get model output
                # get output args.nk+1 times
                if args.guidance_param != 1:
                    cond['y']['scale'] = torch.ones(args.bs, device=dist_util.dev()) * args.guidance_param
                sample_fn = diffusion.p_sample_loop
                Y_list = []
                
                for act_id in range(cfg.vae_specs['n_action']):
                    # 提取当前候选动作的标签
                    label_single = torch.eye(cfg.vae_specs['n_action'], device=device, dtype=dtype)[act_id]
                    label_act = label_single.unsqueeze(0).repeat(bs, 1)  # shape: [bs, n_action]

                    #print("Y shape:", Y.shape)
                    #print("label_act shape:", label_act.shape)
                    # 更新 cond
                    cond_act = copy.deepcopy(cond)
                    cond_act['y']['action'] = label_act
                    Y_output_list = []
                    for _ in range(args.nk):
                        # 模型生成
                        #print("act X shape",cond_act['y']['samp_his'].shape)
                        Y_output= sample_fn(model, Y.shape, clip_denoised=False, model_kwargs=cond_act) 

                        Y_output = Y_output.permute(0, 3, 1, 2).contiguous()  # [B, J, 6, T] → [B, T, J, 6]
                        Y_output = rot6d_to_axisangle(Y_output)  # [B, T, J, 6]→[B, T, J, 3]
                        B,T,J,F = Y_output.shape
                        Y_output = Y_output.permute(1, 0, 2, 3).contiguous().view(T, B, J * F) 

                        # get full sequence
                        Y_output = torch.cat([X, Y_output], dim=0)
                        Y_output_list.append(Y_output)
                    Y_output_cat = torch.cat(Y_output_list, dim=1)

                    Y_list.append(Y_output_cat)
                # get all kinds motion sequence
                # [T_full, bs * n_action * nk, dim]
                Y_r = torch.cat(Y_list, dim=1)

                plot_motion_trajectories(history=X, pred_trans=Y_r,
                         stop_frames=[25],  # 示例停止帧
                         joint_idx=7, dim=1,
                         save_path='trajectory_joint7_yaxis.png')
                #Y_r = sample_fn(model, Y.shape, clip_denoised=False,model_kwargs = cond)       

                # traj is the future motion sequence

                if cfg.dataset == 'babel':
                    index_used = list(range(30)) + list(range(36, 66))
                    X = X[:, :, index_used]

                fn = get_stop_sign(Y_r,args)
                seq_l = torch.where(fn[cfg.t_his:].transpose(0, 1) == 1)[1].cpu().data.numpy()+1
                seq_len.append(seq_l)
                seq_l = seq_l.reshape([-1, args.nk])
                seq_l = torch.where(fn.transpose(0, 1) == 1)[1].cpu().data.numpy() + 1
                #print("sep_l",seq_l.shape)
                #print("bs , n_actions",bs, cfg_classifier.vae_specs['n_action'],args.nk)
                seq_l = seq_l.reshape([bs, cfg_classifier.vae_specs['n_action'], args.nk])
                
                #traj_tmp[:, :, 1:] = traj_tmp[:, :, 1:] * std + mean
                #traj_tmp = rot6d_to_axisangle(traj_tmp)

                x = traj_tmp.cpu().data.numpy()

                if cfg.dataset == 'babel':
                    traj_tmp = torch.clone(traj)
                    index_used = list(range(30)) + list(range(36, 66))
                    traj_tmp[:, :, index_used] = Y_r
                    Y_r = traj_tmp.clone()

                y = Y_r.reshape([-1,bs, cfg_classifier.vae_specs['n_action'], args.nk,Y_r.shape[-1]]).cpu().data.numpy()
                betas = np.zeros(10)
                for ii in range(5):
                    sequence = {'poses': x[:, ii][:seq_gt[ii]], 'betas': betas}
                    key = f'{act}_{ii}_gt'
                    #print("smpl-mode",smpl_model)
                    render_videos_new(sequence, device, cfg.result_dir + f'/{args.mode}', key, w_golbalrot=True, smpl_model=smpl_model)

                    for jj in range(cfg_classifier.vae_specs['n_action']):
                        # test 2
                        for kk in range(2):
                            sequence = {'poses': y[:, ii,jj,kk][:seq_l[ii,jj,kk]], 'betas': betas}
                            key = f'{act}_{ii}_{dataset.act_name[jj]}_{kk}'
                            render_videos_new(sequence, device, cfg.result_dir + f'/{args.mode}', key, w_golbalrot=True, smpl_model=smpl_model)

            print(f">>>> action {act} time used {time.time()-st:.3f}")

if __name__ == '__main__':
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    dist_util.setup_dist(args.device)

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
    max_len=cfg.vae_specs['max_len'] if 'max_len' in cfg.vae_specs else None
    dataset = dataset_cls(args.mode, t_his, t_pred, actions='all', use_vel=cfg.use_vel,
                          acts=cfg.vae_specs['actions'] if 'actions' in cfg.vae_specs else None,
                          max_len=cfg.vae_specs['max_len'] if 'max_len' in cfg.vae_specs else None,
                          min_len=cfg.vae_specs['min_len'] if 'min_len' in cfg.vae_specs else None,
                          is_6d=cfg.vae_specs['is_6d'] if 'is_6d' in cfg.vae_specs else False,
                          data_file=cfg.vae_specs['data_file'] if 'data_file' in cfg.vae_specs else None)
    #print("dataset name",args.dataset)
    #data = get_dataset_loader(name=args.dataset, batch_size=5, num_frames=60,hml_mode='video',cfg=cfg)
    """my model"""
    print("Creating model and diffusion...")
    data = None
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
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
