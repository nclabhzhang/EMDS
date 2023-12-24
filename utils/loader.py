import torch
import random
import numpy as np

from models.ScoreNetwork_A import ScoreNetworkA
from models.ScoreNetwork_X import ScoreNetworkX, ScoreNetworkX_GMH
from models.ScoreNetwork_P import ScoreNetworkP, ScoreNetworkP_GMH
from sde import VPSDE, VESDE, subVPSDE

from losses import get_sde_loss_fn_3d
from solver import get_pc_sampler
from utils.ema import ExponentialMovingAverage


def load_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def load_device():
    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))
    else:
        device = 'cpu'
    return device


def load_model(params):
    params_ = params.copy()
    model_type = params_.pop('model_type', None)

    if model_type == 'ScoreNetworkX':
        model = ScoreNetworkX(**params_)
    elif model_type == 'ScoreNetworkX_GMH':
        model = ScoreNetworkX_GMH(**params_)

    elif model_type == 'ScoreNetworkP':
        model = ScoreNetworkP(**params_)
    elif model_type == 'ScoreNetworkP_GMH':
        model = ScoreNetworkP_GMH(**params_)

    elif model_type == 'ScoreNetworkA':
        model = ScoreNetworkA(**params_)

    else:
        raise ValueError(f"Model Name <{model_type}> is Unknown")
    return model


def load_model_optimizer(params, config_train, device):
    model = load_model(params)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr, weight_decay=config_train.weight_decay)

    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    
    return model, optimizer, scheduler


def load_ema(model, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema


def load_ema_from_ckpt(model, ema_state_dict, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    ema.load_state_dict(ema_state_dict)
    return ema


def load_data(config, get_graph_list=False):
    if config.data.data == 'QM93D':
        from utils.data_loader_mol import dataloader_3d
        return dataloader_3d(config, get_graph_list)
    else:
        raise ValueError(f"Dataset Name <{config.data.data}> is Unknown")


def load_batch(batch, device):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    x_b = batch[0].to(device_id)
    pos_b = batch[1].to(device_id)
    adj_b = batch[2].to(device_id)
    return x_b, pos_b, adj_b


def load_sde(config_sde):
    sde_type = config_sde.type
    beta_min = config_sde.beta_min
    beta_max = config_sde.beta_max
    num_scales = config_sde.num_scales

    if sde_type == 'VP':
        sde = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    elif sde_type == 'VE':
        sde = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_scales)
    elif sde_type == 'subVP':
        sde = subVPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    else:
        raise NotImplementedError(f"SDE class {sde_type} not yet supported.")

    return sde


def load_loss_fn(config):
    reduce_mean = config.train.reduce_mean
    sde_x = load_sde(config.sde.x)
    sde_pos = load_sde(config.sde.pos)
    sde_adj = load_sde(config.sde.adj)
    
    loss_fn = get_sde_loss_fn_3d(sde_x, sde_pos, sde_adj, train=True, reduce_mean=reduce_mean, continuous=True, likelihood_weighting=False, eps=config.train.eps)
    return loss_fn


def load_sampling_fn(config_train, config_module, config_sample, config_data, device):
    sde_x = load_sde(config_train.sde.x)
    sde_pos = load_sde(config_train.sde.pos)
    sde_adj = load_sde(config_train.sde.adj)
    max_node_num = config_train.data.max_node_num

    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device

    get_sampler = get_pc_sampler

    if config_train.data.data in ['QM9', 'ZINC250k', 'QM93D']:
        shape_x = (config_data.batch_size, max_node_num, config_train.data.max_feat_num)
        shape_pos = (config_data.batch_size, max_node_num, 3)
        shape_adj = (config_data.batch_size, max_node_num, max_node_num)
    else:
        shape_x = (config_train.data.batch_size, max_node_num, config_train.data.max_feat_num)
        shape_adj = (config_train.data.batch_size, max_node_num, max_node_num)
        
    sampling_fn = get_sampler(sde_x=sde_x, sde_pos=sde_pos, sde_adj=sde_adj, shape_x=shape_x, shape_pos=shape_pos, shape_adj=shape_adj,
                                predictor=config_module.predictor, corrector=config_module.corrector,
                                snr=config_module.snr, scale_eps=config_module.scale_eps, 
                                n_steps=config_module.n_steps,
                                probability_flow=config_sample.probability_flow, 
                                continuous=True, denoise=config_sample.noise_removal, 
                                eps=config_sample.eps, device=device_id)
    return sampling_fn


def load_model_params(config):
    config_x = config.model_x
    config_p = config.model_pos
    config_a = config.model_adj
    max_feat_num = config.data.max_feat_num

    # if 'GMH' in config_p.name:
    #     params_pos = {'model_type': config_p.name, 'max_feat_num': max_feat_num, 'depth': config_p.depth,
    #                   'nhid': config_p.nhid, 'num_linears': config_a.linears,
    #                   'c_init': config_a.c_init, 'c_hid': config_a.c_hid, 'c_final': config_a.c_final,
    #                   'adim': config_p.adim, 'num_heads': config_a.num_heads, 'conv': config_p.conv,
    #                   'n_layers_egnn': config_p.egnn_gmh.num_layers, 'normalization_factor': config_p.egnn_gmh.normalization_factor,
    #                   'aggregation_method': config_p.egnn_gmh.method}
    # else:
    params_pos = {'model_type': config_p.name, 'max_feat_num': max_feat_num, 'depth': config_p.depth,
                  'nhid': config_p.nhid,
                  'n_layers_egnn': config_p.egnn.num_layers, 'n_hid_egnn': config_p.egnn.nhid,
                  'norm_factor': config_p.egnn.normalization_factor,
                  'aggregation_method': config_p.egnn.method}

    params_x = {'model_type': config_x.name, 'max_feat_num': max_feat_num, 'depth': config_x.depth,
                'nhid': config_x.nhid,
                'n_layers_egnn': config_x.egnn.num_layers, 'n_hid_egnn': config_x.egnn.nhid,
                'norm_factor': config_x.egnn.normalization_factor,
                'aggregation_method': config_x.egnn.method}

    params_adj = {'model_type': config_a.name, 'max_feat_num': max_feat_num, 'max_node_num': config.data.max_node_num,
                  'nhid': config_a.nhid, 'num_layers': config_a.layers, 'num_linears': config_a.linears,
                  'c_init': config_a.c_init, 'c_hid': config_a.c_hid, 'c_final': config_a.c_final,
                  'adim': config_a.adim, 'num_heads': config_a.num_heads, 'conv': config_a.conv,
                  'n_layers_egnn': config_a.egnn.num_layers, 'normalization_factor': config_a.egnn.normalization_factor,
                  'aggregation_method': config_a.egnn.method}
    return params_x, params_pos, params_adj


def load_ckpt(config, device, ts=None, return_ckpt=False):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    ckpt_dict = {}
    if ts is not None:
        config.ckpt = ts
    path = f'./checkpoints/{config.data.data}/{config.ckpt}.pth'
    ckpt = torch.load(path, map_location=device_id)
    print(f'{path} loaded')
    ckpt_dict = {'config': ckpt['model_config'],
                 'params_x': ckpt['params_x'], 'x_state_dict': ckpt['x_state_dict'],
                 'params_pos': ckpt['params_pos'], 'pos_state_dict': ckpt['pos_state_dict'],
                 'params_adj': ckpt['params_adj'], 'adj_state_dict': ckpt['adj_state_dict']}
    if config.sample.use_ema:
        ckpt_dict['ema_x'] = ckpt['ema_x']
        ckpt_dict['ema_pos'] = ckpt['ema_pos']
        ckpt_dict['ema_adj'] = ckpt['ema_adj']
    if return_ckpt:
        ckpt_dict['ckpt'] = ckpt
    return ckpt_dict


def load_model_from_ckpt(params, state_dict, device):
    model = load_model(params)
    if 'module.' in list(state_dict.keys())[0]:
        # strip 'module.' at front; for DataParallel models
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    return model
