from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate
from data_loaders.tensors import grab_collate
from data_loaders.tensors import ntu_collate
from data_loaders.tensors import babel_collate
from data_loaders.tensors import humanact12_collate
from data_loaders.grab.utils.config import Config
import torch
def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    elif name == "grab":
        from data_loaders.grab.utils.dataset_grab_action_transition import DatasetGrab
        return DatasetGrab
    elif name == "ntu":
        from data_loaders.grab.utils.dataset_ntu_act_transition import DatasetNTU
        return DatasetNTU
    elif name == "babel":
        from data_loaders.grab.utils.dataset_babel_action_transition import DatasetBabel
        return DatasetBabel
    elif name == "humanact12kai":
        from data_loaders.grab.utils.dataset_humanact12_act_transition import DatasetACT12
        return DatasetACT12
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    elif name in ["grab"]:
        return grab_collate
    elif name in ["babel"]:
        return babel_collate
    elif name in ["ntu"]:
        return ntu_collate
    elif name in ["humanact12_kai"]:
        return humanact12_collate
    else:
        return all_collate


def get_dataset(name, num_frames, cfg, split='train', hml_mode='train'):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    elif name in ["grab","babel","ntu","humanact12_kai"]:
        #cfg = Config(name ,test=False)
        #mode = hml_mode
        #nz = cfg.nz
        t_his = cfg.t_his
        t_pred = cfg.t_pred
        num_samples=0
        batch_size=0
        if(hml_mode == 'train'):
            num_samples = cfg.num_vae_data_sample
            batch_size=cfg.batch_size
        elif(hml_mode == 'test'):
            num_samples = 50
            batch_size = 10
        elif(hml_mode == 'video'):
            num_samples = 5
            batch_size = 5
            hml_mode = 'test'


        dataset = DATA(hml_mode, t_his, t_pred, actions='all', use_vel=cfg.use_vel,
                          acts=cfg.vae_specs['actions'] if 'actions' in cfg.vae_specs else None,
                          max_len=cfg.vae_specs['max_len'] if 'max_len' in cfg.vae_specs else None,
                          min_len=cfg.vae_specs['min_len'] if 'min_len' in cfg.vae_specs else None,
                          is_6d=cfg.vae_specs['is_6d'] if 'is_6d' in cfg.vae_specs else False,
                            num_samples=num_samples, batch_size=batch_size,
                            is_other_act=cfg.vae_specs['is_other_act'], t_pre_extra=cfg.vae_specs['t_pre_extra'],
                            act_trans_k= cfg.vae_specs['act_trans_k'] if 'act_trans_k'in cfg.vae_specs else 0.08,
                            max_trans_fn= cfg.vae_specs['max_trans_fn'] if 'max_trans_fn'in cfg.vae_specs else 0.08,is_transi=False,
                          data_file=cfg.vae_specs['data_file'] if 'data_file' in cfg.vae_specs else None,
                          w_transi=cfg.vae_specs['w_transi'] if 'w_transi' in cfg.vae_specs else False,
                          )
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    #print("dataset",dataset.shape)
    #print("Dataset Attributes:", dataset.__dict__)


    return dataset


def get_dataset_loader(name, batch_size, num_frames, cfg, split='train', hml_mode='train'):
    if name in ["grab","babel","ntu","humanact12_kai"]:
        cfg = cfg
        dataset = get_dataset(name, num_frames,cfg,split, hml_mode)
        collate = get_collate_fn(name, hml_mode)
        # batch_size作为参数传入
        # batch_size=cfg.batch_size
        #print(1)
        """
        loader = dataset.sampling_generator(num_samples=cfg.num_vae_data_sample, batch_size=cfg.batch_size,
                                           is_other_act=cfg.vae_specs['is_other_act'], t_pre_extra=cfg.vae_specs['t_pre_extra'],
                                           act_trans_k= cfg.vae_specs['act_trans_k'] if 'act_trans_k'
                                                                                        in cfg.vae_specs else 0.08,
                                           max_trans_fn= cfg.vae_specs['max_trans_fn'] if 'max_trans_fn'
                                                                                        in cfg.vae_specs else 0.08,
                                           is_transi=False)
        """
        #print("batch size",batch_size)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=8, drop_last=True,collate_fn=collate)
    else:
        cfg = None
        dataset = get_dataset(name, num_frames, cfg,split, hml_mode)
        collate = get_collate_fn(name, hml_mode)

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=8, drop_last=True, collate_fn=collate
        )

        """
        # 假设 loader 是 DataLoader 实例
        for batch_idx, (motion, cond) in enumerate(loader):
            print(f"Batch {batch_idx + 1}:")
            
            # 打印 motion 的形状和内容
            print("Motion Shape:", motion.shape)
            #print("Motion Data:", motion)  # 可以注释掉这一行防止内容过长
            
            # 打印 cond 字典中的内容和形状
            print("Condition Data (cond):")
            for key, value in cond['y'].items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape {value.shape}")
                    #print(f"  {key} Data:", value)  # 同样可以选择注释以防内容过长
                else:
                    print(f"  {key}: {type(value)}")

            # 仅查看第一个批次，可在此处添加 break 退出循环
            break
        """

    '''
    for batch_idx, (samp, label, fn, fn_mask) in enumerate(loader):
        print(f"Batch {batch_idx + 1}")
        
        # 打印每个变量的形状和部分数据
        print("Sample shape:", samp.shape)
        print("Label shape:", label.shape)
        print("FN shape:", fn.shape)
        print("FN Mask shape:", fn_mask.shape)

        
        # 仅查看一个批次可以添加 break
        break
    '''
    

    """
    First joint sample shape: (64, 24, 3)
    Number of joints: 24
    Joint feature dimension: 3

    Number of Poses: 1190
    Number of Frames per Video: [64, 75, 40, 57, 108]
    Number of Joints: 1190
    Number of Actions: 1190
    Total Number of Actions: 12
    Training Indices: [0, 1, 2, 3, 4]
    Action to Label Mapping: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11}
    Label to Action Mapping: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11}
    Action Classes: {0: 'warm_up', 1: 'walk', 2: 'run', 3: 'jump', 4: 'drink', 5: 'lift_dumbbell', 6: 'sit', 7: 'eat', 8: 'turn steering wheel', 9: 'phone', 10: 'boxing', 11: 'throw'}

    Sample keys: dict_keys(['inp', 'action', 'action_text'])
    Sample shape and data:
    inp: shape torch.Size([25, 6, 60])
    action: <class 'int'>
    action_text: <class 'str'>

    batch shapes:
    Sample 0 shape: torch.Size([25, 6, 60])
    notnone_batches shapes:
    Sample 0 shape: torch.Size([25, 6, 60])

    Sample 63 shape: torch.Size([25, 6, 60])
    databatchTensor shape: torch.Size([64, 25, 6, 60])

    Poses shape: 1190
    Joints shape: 1190
    Actions shape: 1190
    Single pose sample shape: (64, 72)
    Single joints3D sample shape: (64, 24, 3)

    x_start torch.Size([64, 25, 6, 60]) 

    Motion Shape: torch.Size([64, 25, 6, 60])
    Condition Data (cond):
    mask: torch.Size([64, 1, 1, 60])
    lengths: torch.Size([64])
    action: torch.Size([64, 1])
    action_text: <class 'list'>
    """

    return loader