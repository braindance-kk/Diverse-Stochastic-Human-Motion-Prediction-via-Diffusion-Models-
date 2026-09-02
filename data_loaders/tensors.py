import torch
import numpy as np

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]

    '''
    # 假设 databatch 是一个包含张量的列表
    print("databatch shapes:")
    for i, b in enumerate(databatch):
        print(f"Sample {i} shape: {b.shape}")
    '''
    
    databatchTensor = collate_tensors(databatch)
    #print("databatchTensor shape:", databatchTensor.shape)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)

def grab_collate(batch):
    # 分离batch中的motion、fn_gt、fn_mask_gt和label_gt数据
    samp_his = [item[0] for item in batch]
    samp_gt = [item[1] for item in batch]
    fn = [item[2] for item in batch]
    fn_mask = [item[3] for item in batch]
    label = [item[4] for item in batch]

    # 按batch维度拼接所有样本数据
    samp_his = np.concatenate(samp_his, axis=0)
    samp_gt = np.concatenate(samp_gt, axis=0)
    fn = np.concatenate(fn, axis=0)
    fn_mask = np.concatenate(fn_mask, axis=0)
    label = np.concatenate(label, axis=0)
    #samp = np.concatenate([samp_his,samp_gt],axis=1)
    tmp = np.zeros_like(samp_his[:,:,0])
    fn = np.concatenate([tmp,fn],axis=1)
    tmp = np.ones_like(samp_his[:,:,0])

    mask = fn_mask
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(1).unsqueeze(1)


    fn_mask = np.concatenate([tmp,fn_mask],axis=1)

    njoints, nfeats = 55, 3
    #batch_size, seq_len, dim = samp.shape

    batch_size, seq_len_his, dim = samp_his.shape
    samp_his = torch.tensor(samp_his).reshape(batch_size, seq_len_his, njoints, nfeats).permute(0, 2, 3, 1)

    batch_size, seq_len_gt, dim = samp_gt.shape
    samp_gt = torch.tensor(samp_gt).reshape(batch_size, seq_len_gt, njoints, nfeats).permute(0, 2, 3, 1)
    #print("samp dim",dim)
    #print("Original seq_his shape:", dim.shape)
    #samp = torch.tensor(samp).reshape(batch_size, seq_len, njoints, nfeats).permute(0, 2, 3, 1)
    # 假设 fn 的原始形状为 [batch_size, max_seq_len]
    fn_mask = torch.tensor(fn_mask, dtype=torch.float32)
    #mask = fn_mask.unsqueeze(1).unsqueeze(1)  # 先在第 1 维和第 2 维添加维度
    #fn = torch.tensor(fn).unsqueeze(1).unsqueeze(1)
    
    lengths = fn_mask.sum(dim=-1).long()

    
    #print("samp shape",samp.shape)
    #print("fn_mask",fn_mask.shape)

    motion = samp_gt

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

    return motion, cond

def humanact12_collate(batch):
    # 分离batch中的motion、fn_gt、fn_mask_gt和label_gt数据
    samp_his = [item[0] for item in batch]
    samp_gt = [item[1] for item in batch]
    fn = [item[2] for item in batch]
    fn_mask = [item[3] for item in batch]
    label = [item[4] for item in batch]

    # 按batch维度拼接所有样本数据
    samp_his = np.concatenate(samp_his, axis=0)
    samp_gt = np.concatenate(samp_gt, axis=0)
    fn = np.concatenate(fn, axis=0)
    fn_mask = np.concatenate(fn_mask, axis=0)
    label = np.concatenate(label, axis=0)
    samp = np.concatenate([samp_his,samp_gt],axis=1)
    tmp = np.zeros_like(samp_his[:,:,0])
    fn = np.concatenate([tmp,fn],axis=1)
    tmp = np.ones_like(samp_his[:,:,0])
    fn_mask = np.concatenate([tmp,fn_mask],axis=1)

    njoints, nfeats = 24, 3
    batch_size, seq_len, dim = samp.shape
    #print("Original seq_his shape:", dim.shape)
    samp = torch.tensor(samp).reshape(batch_size, seq_len, njoints, nfeats).permute(0, 2, 3, 1)
    # 假设 fn 的原始形状为 [batch_size, max_seq_len]
    lengths = fn_mask.sum(dim=-1).long() 
    mask = torch.tensor(fn_mask).unsqueeze(1).unsqueeze(1)  # 先在第 1 维和第 2 维添加维度
    #print("samp shape",samp.shape)
    #print("fn_mask",fn_mask.shape)
    # fn 的形状现在是 [batch_size, 1, 1, max_seq_len]
    '''
    samp = torch.tensor(samp, dtype=torch.float32) 
    fn = torch.tensor(fn, dtype=torch.float32)
    fn_mask = torch.tensor(fn_mask, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.float32)
    '''
    
    motion = samp
    #print("motion",motion.shape)
    #print(motion)
    # 在 cond 中存储 fn, fn_mask, 和 label 数据
    cond = {
        'y': {
            'mask': mask,
            'lengths': lengths,
            'action': label,
            'fn':fn,
            'fn_mask':fn_mask

        }
    }

    return motion, cond

def babel_collate(batch):
    # 分离batch中的motion、fn_gt、fn_mask_gt和label_gt数据
    samp_his = [item[0] for item in batch]
    samp_gt = [item[1] for item in batch]
    fn = [item[2] for item in batch]
    fn_mask = [item[3] for item in batch]
    label = [item[4] for item in batch]

    # 按batch维度拼接所有样本数据
    samp_his = np.concatenate(samp_his, axis=0)
    samp_gt = np.concatenate(samp_gt, axis=0)
    fn = np.concatenate(fn, axis=0)
    fn_mask = np.concatenate(fn_mask, axis=0)
    label = np.concatenate(label, axis=0)
    samp = np.concatenate([samp_his,samp_gt],axis=1)
    tmp = np.zeros_like(samp_his[:,:,0])
    fn = np.concatenate([tmp,fn],axis=1)
    tmp = np.ones_like(samp_his[:,:,0])
    fn_mask = np.concatenate([tmp,fn_mask],axis=1)

    njoints, nfeats = 52, 3
    batch_size, seq_len, dim = samp.shape
    #print("Original seq_his shape:", dim.shape)
    samp = torch.tensor(samp).reshape(batch_size, seq_len, njoints, nfeats).permute(0, 2, 3, 1)
    # 假设 fn 的原始形状为 [batch_size, max_seq_len]
    lengths = fn_mask.sum(dim=-1).long() 
    mask = torch.tensor(fn_mask).unsqueeze(1).unsqueeze(1)  # 先在第 1 维和第 2 维添加维度
    #print("samp shape",samp.shape)
    #print("fn_mask",fn_mask.shape)
    # fn 的形状现在是 [batch_size, 1, 1, max_seq_len]
    '''
    samp = torch.tensor(samp, dtype=torch.float32) 
    fn = torch.tensor(fn, dtype=torch.float32)
    fn_mask = torch.tensor(fn_mask, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.float32)
    '''
    
    motion = samp
    #print("motion",motion.shape)
    #print(motion)
    # 在 cond 中存储 fn, fn_mask, 和 label 数据
    cond = {
        'y': {
            'mask': mask,
            'lengths': lengths,
            'action': label,
            'fn':fn,
            'fn_mask':fn_mask

        }
    }

    return motion, cond

def ntu_collate(batch):
    # 分离batch中的motion、fn_gt、fn_mask_gt和label_gt数据
    samp_his = [item[0] for item in batch]
    samp_gt = [item[1] for item in batch]
    fn = [item[2] for item in batch]
    fn_mask = [item[3] for item in batch]
    label = [item[4] for item in batch]

    # 按batch维度拼接所有样本数据
    samp_his = np.concatenate(samp_his, axis=0)
    samp_gt = np.concatenate(samp_gt, axis=0)
    fn = np.concatenate(fn, axis=0)
    fn_mask = np.concatenate(fn_mask, axis=0)
    label = np.concatenate(label, axis=0)
    samp = np.concatenate([samp_his,samp_gt],axis=1)
    tmp = np.zeros_like(samp_his[:,:,0])
    fn = np.concatenate([tmp,fn],axis=1)
    tmp = np.ones_like(samp_his[:,:,0])
    fn_mask = np.concatenate([tmp,fn_mask],axis=1)

    njoints, nfeats = 24, 3
    batch_size, seq_len, dim = samp.shape
    #print("Original seq_his shape:", dim.shape)
    samp = torch.tensor(samp).reshape(batch_size, seq_len, njoints, nfeats).permute(0, 2, 3, 1)
    # 假设 fn 的原始形状为 [batch_size, max_seq_len]
    lengths = fn_mask.sum(dim=-1).long() 
    mask = torch.tensor(fn_mask).unsqueeze(1).unsqueeze(1)  # 先在第 1 维和第 2 维添加维度
    #print("samp shape",samp.shape)
    #print("fn_mask",fn_mask.shape)
    # fn 的形状现在是 [batch_size, 1, 1, max_seq_len]
    '''
    samp = torch.tensor(samp, dtype=torch.float32) 
    fn = torch.tensor(fn, dtype=torch.float32)
    fn_mask = torch.tensor(fn_mask, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.float32)
    '''
    
    motion = samp
    #print("motion",motion.shape)
    #print(motion)
    # 在 cond 中存储 fn, fn_mask, 和 label 数据
    cond = {
        'y': {
            'mask': mask,
            'lengths': lengths,
            'action': label,
            'fn':fn,
            'fn_mask':fn_mask

        }
    }

    return motion, cond

