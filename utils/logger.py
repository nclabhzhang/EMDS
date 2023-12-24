import os

class Logger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def log(self, str, verbose=True):
        if self.lock:
            self.lock.acquire()
        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + '\n')
        except Exception as e:
            print(e)
        if self.lock:
            self.lock.release()
        if verbose:
            print(str)


def set_log(config, is_train=True):
    data = config.data.data
    exp_name = config.train.name

    log_folder_name = os.path.join(*[data, exp_name])
    root = 'logs_train' if is_train else 'logs_sample'
    if not(os.path.isdir(f'./{root}/{log_folder_name}')):
        os.makedirs(os.path.join(f'./{root}/{log_folder_name}'))
    log_dir = os.path.join(f'./{root}/{log_folder_name}/')

    if not(os.path.isdir(f'./checkpoints/{data}')) and is_train:
        os.makedirs(os.path.join(f'./checkpoints/{data}'))
    ckpt_dir = os.path.join(f'./checkpoints/{data}/')

    print('-'*100)
    print("Make Directory {} in Logs".format(log_folder_name))

    return log_folder_name, log_dir, ckpt_dir


def check_log(log_folder_name, log_name):
    return os.path.isfile(f'./logs_sample/{log_folder_name}/{log_name}.log')


def data_log(logger, config):
    logger.log(f'[{config.data.data}]  init={config.data.init} ({config.data.max_feat_num})  seed={config.seed}  batch_size={config.data.batch_size}')


def sde_log(logger, config_sde):
    sde_x = config_sde.x
    sde_pos = config_sde.pos
    sde_adj = config_sde.adj
    logger.log(f'(x:{sde_x.type})=({sde_x.beta_min:.2f}, {sde_x.beta_max:.2f}) N={sde_x.num_scales} '
               f'(pos:{sde_pos.type})=({sde_pos.beta_min:.2f}, {sde_pos.beta_max:.2f}) N={sde_pos.num_scales} ' 
                f'(adj:{sde_adj.type})=({sde_adj.beta_min:.2f}, {sde_adj.beta_max:.2f}) N={sde_adj.num_scales}')


def model_log(logger, config):
    config_x = config.model_x
    config_p = config.model_pos
    config_a = config.model_adj
    if 'GMH' in config_p.name:
        model_log = f'({config_x.name})+({config_p.name} + {config_a.name}={config_a.conv},{config_a.num_heads}) : ' \
                    f'depth_x={config_x.depth} nhid_x={config_x.nhid} ' \
                    f'att_layers(p)={config_p.depth} nhid_p={config_p.nhid} att_dim(p)={config_p.adim}' \
                    f'att_layers(a)={config_a.layers} att_dim(a)={config_a.adim} nhid_a={config_a.nhid}' \
                    f'linears={config_a.linears} c=({config_a.c_init} {config_a.c_hid} {config_a.c_final})'
        egnn_log = f'egnn_layer(x,p,a)=({config_x.egnn.num_layers},{config_p.egnn_gmh.num_layers},{config_a.egnn.num_layers}) ' \
                   f'egnn_nid(x,p,a)=({config_x.egnn.nhid},{config_p.egnn_gmh.nhid},{config_a.egnn.nhid}) ' \
                   f'norm_factor(x,p,a)=({config_x.egnn.normalization_factor},{config_p.egnn_gmh.normalization_factor},{config_a.egnn.normalization_factor}) ' \
                   f'aggregation(x,p,a)=({config_x.egnn.method},{config_p.egnn_gmh.method},{config_a.egnn.method}) '
    else:
        model_log = f'({config_x.name})+({config_p.name})+({config_a.name}={config_a.conv},{config_a.num_heads}) : '\
                    f'depth_x={config_x.depth} nhid_x={config_x.nhid} '\
                    f'depth_p={config_p.depth} nhid_p={config_p.nhid} '\
                    f'att_layers(a)={config_a.layers} att_dim(a)={config_a.adim} nhid_a={config_a.nhid}' \
                    f'linears={config_a.linears} c=({config_a.c_init} {config_a.c_hid} {config_a.c_final})'

        egnn_log = f'egnn_layer(x,p,a)=({config_x.egnn.num_layers},{config_p.egnn.num_layers},{config_a.egnn.num_layers}) '\
                   f'egnn_nid(x,p,a)=({config_x.egnn.nhid},{config_p.egnn.nhid},{config_a.egnn.nhid}) '\
                   f'norm_factor(x,p,a)=({config_x.egnn.normalization_factor},{config_p.egnn.normalization_factor},{config_a.egnn.normalization_factor}) '\
                   f'aggregation(x,p,a)=({config_x.egnn.method},{config_p.egnn.method},{config_a.egnn.method}) '

    logger.log(model_log)
    logger.log(egnn_log)


def start_log(logger, config):
    logger.log('-' * 30)
    data_log(logger, config)
    logger.log('-' * 30)


def train_log(logger, config):
    logger.log(f'lr={config.train.lr} schedule={config.train.lr_schedule} ema={config.train.ema} '
               f'epochs={config.train.num_epochs} reduce={config.train.reduce_mean} eps={config.train.eps}')
    model_log(logger, config)
    sde_log(logger, config.sde)
    logger.log('-' * 30)


def sample_log(logger, config):
    sample_log = f"({config.sampler.predictor})+({config.sampler.corrector}): " \
                 f"eps={config.sample.eps} denoise={config.sample.noise_removal} " \
                 f"ema={config.sample.use_ema} "
    if config.sampler.corrector == 'Langevin':
        sample_log += f'|| snr={config.sampler.snr} seps={config.sampler.scale_eps} '\
                        f'n_steps={config.sampler.n_steps} '
    logger.log(sample_log)
    logger.log('-' * 30)