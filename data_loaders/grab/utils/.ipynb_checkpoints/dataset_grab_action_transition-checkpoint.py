import numpy as np
import os
from data_loaders.grab.utils.dataset import Dataset
# from motion_pred.utils.skeleton import Skeleton
import torch
import kornia.geometry as torchgeometry

from utils.rotation_conversions import axis_angle_to_matrix
from utils.rotation_conversions import matrix_to_rotation_6d

import utils.rotation_conversions as geometry

class DatasetGrab(Dataset):
    def __init__(self, mode, t_his=25, t_pred=100, actions='all', use_vel=False,
                 is_6d=True, num_samples=1000, batch_size=128,is_other_act=False,t_pre_extra=50,
                 act_trans_k=0.08,max_trans_fn=0.08,is_transi=False,**kwargs):
        self.use_vel = use_vel
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.is_other_act = is_other_act
        self.t_pre_extra = t_pre_extra
        self.act_trans_k = act_trans_k
        self.max_trans_fn = max_trans_fn
        self.is_transi = is_transi

        if 'acts' in kwargs.keys() and kwargs['acts'] is not None:
            self.act_name = np.array(kwargs['acts'])
        else:
            self.act_naåme = np.array(["pass", "lift", "inspect", "drink"])
        if 'max_len' in kwargs.keys() and kwargs['max_len'] is not None:
            self.max_len = np.array(kwargs['max_len'])
        else:
            self.max_len = 1000

        if 'min_len' in kwargs.keys() and kwargs['min_len'] is not None:
            self.min_len = np.array(kwargs['min_len'])
        else:
            self.min_len = 100

        self.mode = mode

        if 'data_file' in kwargs.keys() and kwargs['data_file'] is not None:
            self.data_file = kwargs['data_file'].format(self.mode)
        else:
            self.data_file = os.path.join('./data', f'grab_200_1000_wact_candi_{self.mode}.npz')
            
        self.std, self.mean = None, None
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        self.traj_dim = 165
        self.normalized = False
        # iterator specific
        self.sample_ind = None
        self.is_6d = is_6d
        if is_6d:
            #default *2
            self.traj_dim = self.traj_dim
            
        self.process_data()
        #self.normalize_data()
        #self.std, self.mean = None, None
        self.data_len = sum([len(seq) for seq in self.data.values()])

    def axis_angle_to_rot6d(self,axis_angle):
        rotmat = geometry.axis_angle_to_matrix(axis_angle)  # (..., 3, 3)
        rot6d = geometry.matrix_to_rotation_6d(rotmat)      # (..., 6)
        return rot6d
    
    def process_data(self):
        print(f'load data from {self.data_file}')
        data_o = np.load(self.data_file, allow_pickle=True)
        data_f = data_o['data'].item()
        data_cand = data_o['data_cand'].item()
        if self.is_6d:
            data_f_6d = {}
            for key in data_f.keys():
                if key not in data_f_6d.keys():
                    data_f_6d[key] = []
                data_tmp = data_f[key]
                for i, seq in enumerate(data_tmp):
                    fn = seq.shape[0]
                    #print("Axis-angle max before:", np.abs(seq).max())
                    # 假定 seq 原本 shape 为 [fn, ..., 3]，此处将其 reshape 成 [fn * num_joints, 3]
                    seq = seq.reshape([fn, -1, 3]).reshape([-1, 3])
                    # 注意：确保 seq 的 dtype 为 float，并转换为 torch tensor
                    seq_tensor = torch.from_numpy(seq).float()
                    # 使用 axis_angle_to_rot6d 进行转换，输出 shape 为 [fn*num_joints, 6]
                    rot6d = self.axis_angle_to_rot6d(seq_tensor)
                    # 恢复回原始帧的维度；例如，将其 reshape 为 [fn, -1, 6]，如果需要再 reshape 成 [fn, -1]
                    rot6d = rot6d.reshape([fn, -1, 6]).reshape([fn, -1])
                    data_f_6d[key].append(rot6d.data.numpy())
            data_f = data_f_6d

        self.data = data_f
        self.data_cand = data_cand

    def normalize_data(self, mean=None, std=None):
        print("trying normalization")
        if mean is None:
            all_seq = []
            for data_s in self.data.values():
                for seq in data_s:
                    all_seq.append(seq[:, 1:])
            all_seq = np.concatenate(all_seq)
            self.mean = all_seq.mean(axis=0)
            self.std = all_seq.std(axis=0)
        else:
            self.mean = mean
            self.std = std
        for act, seq_list in self.data.items():
            for i in range(len(seq_list)):
                self.data[act][i][:, 1:] = (self.data[act][i][:, 1:] - self.mean) / (self.std + 1e-8)



        """
        # 保存 mean 和 std
        save_dir = os.path.join('.', 'dataset')
        os.makedirs(save_dir, exist_ok=True)

        mean_filename = f'grab_{self.mode}_mean.npy'
        std_filename = f'grab_{self.mode}_std.npy'

        np.save(os.path.join(save_dir, mean_filename), self.mean)
        np.save(os.path.join(save_dir, std_filename), self.std)

        print(f"Saved mean to: {os.path.join(save_dir, mean_filename)}")
        print(f"Saved std to: {os.path.join(save_dir, std_filename)}")
        """
        

        self.normalized = True

    
    def sample(self,action=None, is_other_act=False,t_pre_extra=0, k=0.08, max_trans_fn=25):
        if action is None:
            action = np.random.choice(self.act_name)
        #action = "pass"
        max_seq_len = self.max_len.item() - self.t_his + t_pre_extra
        seq = self.data[action]
        idx = np.random.randint(0, len(seq))
        seq = seq[idx]
        fn = seq.shape[0]
        if fn // 10 > self.t_his:
            fr_start = np.random.randint(0, fn // 10 - self.t_his)
            seq = seq[fr_start:]
            fn = seq.shape[0]

        seq_his = seq[:self.t_his][None,:,:]
        seq_tmp = seq[self.t_his:]
        fn = seq_tmp.shape[0]
        seq_gt = np.zeros([1, max_seq_len, seq.shape[-1]])
        seq_gt[0, :fn] = seq_tmp
        seq_gt[0,fn:] = seq_tmp[-1:]
        fn_gt = np.zeros([1, max_seq_len])
        # 在当前目标序列的最后一个有效时间步（fn - 1）位置上设置为 1。
        fn_gt[:, fn - 1] = 1
        fn_mask_gt = np.zeros([1, max_seq_len])
        # 从第一个时间步到目标序列的最后一个时间步（包括额外的预测时间 t_pre_extra）的位置上设置为 1，表示这些时间步是有效的。
        fn_mask_gt[:, :fn+t_pre_extra] = 1
        label_gt = np.zeros(len(self.act_name))
        tmp = str.lower(action.split(' ')[0])
        label_gt[np.where(tmp == self.act_name)[0]] = 1
        label_gt = label_gt[None,:]

        # randomly find future sequences of other actions
        if is_other_act:
            #print("trying find future sequence")
            # k = 0.08
            # max_trans_fn = 25
            seq_last = seq_his[0,-1:]
            seq_others = []
            fn_others = []
            fn_mask_others = []
            label_others = []
            cand_seqs = self.data_cand[f'{action}_{idx}']

            act_names = np.random.choice(self.act_name, len(self.act_name))
            for act in act_names:
                cand = cand_seqs[act]
                if len(cand)<=0:
                    continue
                for _ in range(10):
                    cand_idx = np.random.choice(cand, 1)[0]
                    cand_tmp = self.data[act][cand_idx]
                    cand_fn = cand_tmp.shape[0]
                    cand_his = cand_tmp[:max(cand_fn//10,25)]
                    dd = np.linalg.norm(cand_his-seq_last, axis=1)
                    cand_tmp = cand_tmp[np.where(dd==dd.min())[0][0]:]
                    cand_fn = cand_tmp.shape[0]
                    skip_fn = min(int(dd.min()//k + 1), max_trans_fn)
                    if cand_fn + skip_fn+self.t_his > self.max_len:
                        continue
                    # cand_tmp = np.copy(cand[[-1] * (self.max_len.item()-self.t_his)])[None, :, :]
                    cand_tt = np.zeros([1, max_seq_len, seq.shape[-1]])
                    cand_tt[0, :skip_fn] = cand_tmp[:1]
                    cand_tt[0, skip_fn:cand_fn+skip_fn] = cand_tmp
                    cand_tt[0,cand_fn+skip_fn:] = cand_tmp[-1:]
                    fn_tmp = np.zeros([1, max_seq_len])
                    fn_tmp[:, cand_fn+skip_fn-1] = 1
                    fn_mask_tmp = np.zeros([1, max_seq_len])
                    fn_mask_tmp[:, skip_fn:cand_fn+skip_fn+t_pre_extra] = 1
                    cand_lab = np.zeros(len(self.act_name))
                    cand_lab[np.where(act == self.act_name)[0]] = 1
                    seq_others.append(cand_tt)
                    fn_others.append(fn_tmp)
                    fn_mask_others.append(fn_mask_tmp)
                    label_others.append(cand_lab[None,:])
                    break
                break
                
            if len(seq_others) > 0:
                seq_others = np.concatenate(seq_others,axis=0)
                fn_others = np.concatenate(fn_others,axis=0)
                fn_mask_others = np.concatenate(fn_mask_others,axis=0)
                label_others = np.concatenate(label_others,axis=0)

                seq_his = seq_his[[0]*(seq_others.shape[0]+1)]
                seq_gt = np.concatenate([seq_gt,seq_others], axis=0)
                fn_gt = np.concatenate([fn_gt,fn_others], axis=0)
                fn_mask_gt = np.concatenate([fn_mask_gt,fn_mask_others], axis=0)
                label_gt = np.concatenate([label_gt, label_others], axis=0)
        #print("seq_his",seq_his.shape)
        #print("action:", action, "seq idx:", idx)

        # ==== Rot6D conversion ====
        """
        seq_his_tensor = torch.from_numpy(seq_his)  # [B, T, 165]
        seq_gt_tensor = torch.from_numpy(seq_gt)

        B_his, T_his, D = seq_his_tensor.shape
        B_gt, T_gt, _ = seq_gt_tensor.shape

        seq_his_reshaped = seq_his_tensor.view(B_his, T_his, 55, 3)
        seq_gt_reshaped = seq_gt_tensor.view(B_gt, T_gt, 55, 3)

        seq_his_rotmat = geometry.axis_angle_to_matrix(seq_his_reshaped)
        seq_gt_rotmat = geometry.axis_angle_to_matrix(seq_gt_reshaped)

        seq_his_rot6d = geometry.matrix_to_rotation_6d(seq_his_rotmat).view(B_his, T_his, -1).numpy()
        seq_gt_rot6d = geometry.matrix_to_rotation_6d(seq_gt_rotmat).view(B_gt, T_gt, -1).numpy()


        return seq_his_rot6d, seq_gt_rot6d, fn_gt, fn_mask_gt, label_gt
        """
        #print("seq-his",seq_his.shape)
        return seq_his,seq_gt,fn_gt,fn_mask_gt,label_gt
    


    def sampling_generator(self, num_samples=1000, batch_size=8,act=None,is_other_act=False,t_pre_extra=0,
                           act_trans_k=0.08, max_trans_fn=25, is_transi=False):
        for i in range(num_samples // batch_size):
            samp_his = []
            samp_gt = []
            fn = []
            fn_mask = []
            label = []
            for i in range(batch_size):
                #
                seq_his, seq_gt, fn_gt, fn_mask_gt, label_gt = self.sample(action=act,is_other_act=is_other_act,
                                                                           t_pre_extra=t_pre_extra,
                                                                           k=act_trans_k,max_trans_fn=max_trans_fn)
                samp_his.append(seq_his)
                samp_gt.append(seq_gt)
                fn.append(fn_gt)
                fn_mask.append(fn_mask_gt)
                label.append(label_gt)
            samp_his = np.concatenate(samp_his, axis=0)
            samp_gt = np.concatenate(samp_gt, axis=0)
            fn = np.concatenate(fn, axis=0)
            fn_mask = np.concatenate(fn_mask, axis=0)
            label = np.concatenate(label, axis=0)
            samp = np.concatenate([samp_his,samp_gt],axis=1)
            tmp = np.zeros_like(samp_his[:,:,0])
            fn = np.concatenate([tmp,fn],axis=1)
            tmp = np.ones_like(samp_his[:,:,0])

            mask = fn_mask
            mask = torch.tensor(mask, dtype=torch.float64).unsqueeze(1).unsqueeze(1)

            lengths = np.sum(fn_mask, axis=-1).astype(np.int64)

            fn_mask = np.concatenate([tmp,fn_mask],axis=1)
            njoints, nfeats = 55, 6
            #batch_size, seq_len, dim = samp.shape

            batch_size_his, seq_len_his, dim = samp_his.shape
            samp_his = torch.tensor(samp_his).reshape(batch_size_his, seq_len_his, njoints, nfeats).permute(0, 2, 3, 1)

            batch_size_gt, seq_len_gt, dim = samp_gt.shape
            samp_gt = torch.tensor(samp_gt).reshape(batch_size_gt, seq_len_gt, njoints, nfeats).permute(0, 2, 3, 1)

            cond = {
                'y': {
                    'mask': mask,
                    'lengths': lengths,
                    'action': label,
                    'fn':fn,
                    'fn_mask':fn_mask,
                    'samp_his':samp_his

                }
            }
            motion = samp_gt
            #print("motion",motion.shape)
            yield motion,label, fn, fn_mask, cond


    def iter_generator(self, step=25):
        for data_s in self.data.values():
            for seq in data_s.values():
                seq_len = seq.shape[0]
                for i in range(0, seq_len - self.t_total, step):
                    traj = seq[None, i: i + self.t_total]
                    yield traj / 1000.


if __name__ == '__main__':
    np.random.seed(0)
    actions = {'WalkDog'}
    dataset = DatasetGrab('train')
    generator = dataset.sampling_generator()
    # dataset.normalize_data()
    # generator = dataset.iter_generator()
    for data, action, fn in generator:
        print(data.shape)
