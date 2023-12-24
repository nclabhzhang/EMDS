import os
import torch
import torch.nn.functional as F
import numpy as np
from utils.data_loader_mol import load_mol_3d, get_3d_transform_fn


# -------- Mask batch of node features with 0-1 flags tensor --------
def mask_x(x, flags):
    """
        :param x:  B x N x k, k is the num of atom types, eg: 5 for QM93D
        :param flags: B x N
    """
    if flags is None:
        flags = torch.ones((x.shape[0], x.shape[1]), device=x.device)
    return x * flags[:,:,None]

# -------- Mask batch of 3d coordinate matrices with 0-1 flags tensor --------
def mask_pos(pos, flags):
    """
        :param pos:  B x N x 3
        :param flags: B x N
    """
    if flags is None:
        flags = torch.ones((pos.shape[0], pos.shape[1]), device=pos.device)
    return pos * flags[:,:,None]  # need to adjust dim!!!


# -------- Mask batch of adjacency matrices with 0-1 flags tensor --------
def mask_adjs(adjs, flags):
    """
    :param adjs:  B x N x N or B x C x N x N
    :param flags: B x N
    :return:
    """
    if flags is None:
        flags = torch.ones((adjs.shape[0], adjs.shape[-1]), device=adjs.device)

    if len(adjs.shape) == 4:
        flags = flags.unsqueeze(1)  # B x 1 x N
    adjs = adjs * flags.unsqueeze(-1)
    adjs = adjs * flags.unsqueeze(-2)
    return adjs

def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)
    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x



# -------- Create flags tensor from graph dataset --------
def node_flags(adj, eps=1e-5):
    flags = torch.abs(adj).sum(-1).gt(eps).to(dtype=torch.float32)
    if len(flags.shape) == 3:
        flags = flags[:,0,:]
    return flags


# -------- Sample initial flags tensor from the training set -------
def init_flags_3d(config, batch_size=None):
    if batch_size is None:
        batch_size = config.data.batch_size
    max_node_num = config.data.max_node_num

    mols = load_mol_3d(os.path.join(config.data.dir, f'{config.data.data.lower()}_kekulized.npz'))

    split_idxs = np.load(os.path.join(config.data.dir, 'split.npz'))
    split_dict = {
        'train': split_idxs['train_idx'].tolist(),
        'valid': split_idxs['val_idx'].tolist(),
        'test': split_idxs['test_idx'].tolist()
    }

    train_adj = torch.cat([torch.tensor(mols[i][2]).unsqueeze(0) for i in split_dict['train']], 0)
    # valid_adj = torch.cat([torch.tensor(mols[i][2]).unsqueeze(0) for i in split_dict['valid']], 0)
    test_adj = torch.cat([torch.tensor(mols[i][2]).unsqueeze(0) for i in split_dict['test']], 0)

    print(f'Number of training mols: {len(split_dict["train"])} | Number of test mols: {len(split_dict["test"])}')

    idx = np.random.randint(0, len(train_adj), batch_size)
    flags = node_flags(train_adj[idx])

    return flags


# -------- Generate noise --------
def gen_noise(x, flags, sym=True):
    z = torch.randn_like(x)
    if sym:  # for generating adj noise
        z = z.triu(1)
        z = z + z.transpose(-1, -2)
        z = mask_adjs(z, flags)
    else:
        z = mask_x(z, flags)
    return z


# -------- Create higher order adjacency matrices --------
def pow_tensor(x, cnum):
    # x : B x N x N
    x_ = x.clone()
    xc = [x.unsqueeze(1)]
    for _ in range(cnum-1):
        x_ = torch.bmm(x_, x)
        xc.append(x_.unsqueeze(1))
    xc = torch.cat(xc, dim=1)

    return xc   # [x, x^2, x^3, ..., x^cnum]


def node_feature_to_matrix(x):
    """
    :param x:  BS x N x F
    :return:
    x_pair: BS x N x N x 2F
    """
    x_b = x.unsqueeze(-2).expand(x.size(0), x.size(1), x.size(1), -1)  # BS x N x N x F
    x_pair = torch.cat([x_b, x_b.transpose(1, 2)], dim=-1)  # BS x N x N x 2F

    return x_pair
