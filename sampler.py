import os
import torch
import numpy as np
import json

from utils.logger import Logger, set_log, start_log, train_log, sample_log, check_log
from utils.loader import load_ckpt, load_data, load_seed, load_device, load_model_from_ckpt, load_sampling_fn
from utils.graph_utils import init_flags_3d
from utils.mol_utils import gen_mol, gen_sigle_mol, mols_to_smiles
from utils.eval_validity_utils import xyz2mol
from utils.loader import load_model_from_ckpt, load_ckpt, load_device, load_batch, load_sde, load_seed
from nll import compute_NLL


# -------- Sampler for molecule generation tasks --------
class Sampler_mol(object):
    def __init__(self, config):
        self.config = config
        self.device = load_device()

    def sample(self):
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)
        self.configt = self.ckpt_dict['config']

        load_seed(self.config.seed)

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.ckpt}-sample"
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')

        if not check_log(self.log_folder_name, self.log_name):
            start_log(logger, self.configt)
            train_log(logger, self.configt)
        sample_log(logger, self.config)

        # -------- Load models --------
        self.model_x = load_model_from_ckpt(self.ckpt_dict['params_x'], self.ckpt_dict['x_state_dict'], self.device)
        self.model_pos = load_model_from_ckpt(self.ckpt_dict['params_pos'], self.ckpt_dict['pos_state_dict'], self.device)
        self.model_adj = load_model_from_ckpt(self.ckpt_dict['params_adj'], self.ckpt_dict['adj_state_dict'], self.device)

        self.sampling_fn = load_sampling_fn(self.configt, self.config.sampler, self.config.sample, self.config.data, self.device)

        # -------- Generate samples --------
        logger.log(f'GEN SEED: {self.config.sample.seed}')
        load_seed(self.config.sample.seed)

        self.init_flags = init_flags_3d(self.configt, self.config.data.batch_size).to(self.device[0])
        x, pos, adj, _ = self.sampling_fn(self.model_x, self.model_pos, self.model_adj, self.init_flags)

        x = torch.where(x > 0.5, 1, 0)
        x = x.cpu().numpy()
        x = np.argmax(x, axis=2)
        x = np.where(x > 0, x + 5, 1)
        x = x * self.init_flags.cpu().numpy()
        pos = pos.cpu().numpy()

        samples = {'x': [],
                   'pos': [],
                   'adj': [],
                   'val': []
                   }
        for i in range(x.shape[0]):
            n_atoms = np.argmin(x[i])
            cur_x, cur_pos = x[i][:n_atoms].astype(int), pos[i][:n_atoms]
            cur_a, valid = xyz2mol(cur_x, cur_pos)
            samples['x'].append(cur_x)
            samples['pos'].append(cur_pos)
            samples['adj'].append(cur_a)
            samples['val'].append(1 if valid else 0)

        # -------- Evaluation --------
        num_atom = 0
        num_instable_atom = 0
        num_valid = 0
        num_stable_mol = 0
        val_mols = []
        for i in range(len(samples['x'])):
            cur_x, cur_pos, cur_adj = samples['x'][i], samples['pos'][i], samples['adj'][i]
            is_vaild = samples['val'][i]

            valency = np.sum(cur_adj, axis=1)
            flag = True
            for j in range(len(cur_x)):
                num_atom += 1
                if cur_x[j] == 1 and valency[j] == 1:
                    continue
                elif cur_x[j] == 6 and valency[j] == 4:
                    continue
                elif cur_x[j] == 7 and valency[j] == 3:
                    continue
                elif cur_x[j] == 8 and valency[j] == 2:
                    continue
                elif cur_x[j] == 9 and valency[j] == 1:
                    continue
                else:
                    num_instable_atom += 1
                    flag = False
            if flag:
                num_stable_mol += 1

            if is_vaild == 1:
                num_valid += 1
                gen = gen_sigle_mol(cur_x, cur_adj)
                if gen_mol is not None:
                    val_mols.append(gen)

        val_smiles = mols_to_smiles(val_mols)
        num_uniqueness = len(set(val_smiles))

        print("Valid Ratio: {}/{} = {:.2f}%".format(num_valid, self.config.data.batch_size, num_valid / self.config.data.batch_size * 100))
        print("Valid and Unique Ratio: {}/{} = {:.2f}%".format(num_uniqueness, self.config.data.batch_size, num_uniqueness / self.config.data.batch_size * 100))

        print("Atom Stable Ratio: {}/{} = {:.2f}%".format((num_atom - num_instable_atom), num_atom, (num_atom - num_instable_atom) / num_atom * 100))
        print("Mol Stable Ratio: {}/{} = {:.2f}%".format(num_stable_mol, self.config.data.batch_size, num_stable_mol / self.config.data.batch_size * 100))

        # -------- NLL --------
        _, test_loader = load_data(self.config)
        sde_x = load_sde(self.configt.sde.x)
        sde_pos = load_sde(self.configt.sde.pos)
        sde_adj = load_sde(self.configt.sde.adj)

        nll_epoch = 0
        n_samples = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, pos, adj = load_batch(data, self.device)
                batch_size = x.size(0)
                nll = compute_NLL(x, pos, adj, self.model_x, self.model_pos, self.model_adj, sde_x, sde_pos, sde_adj)
                nll_epoch += nll.item() * batch_size
                n_samples += batch_size

        nll = nll_epoch / n_samples
        print("NLL: {:.2f}".format(nll))
        # print("NLL: {}/{} = {:.2f}".format(nll_epoch, n_samples, nll))

        # -------- Save generated molecules --------
        if not os.path.exists('./sample_mols'):
            os.makedirs('./sample_mols')
        samples_save = self.log_name.replace('/', '')[:-7] + '_' + f'{self.config.data.batch_size}'
        save_data = []
        for i in range(len(samples['x'])):
            cur_mol = {}
            cur_mol['atom'] = samples['x'][i].tolist()
            cur_mol['pos'] = samples['pos'][i].tolist()
            cur_mol['bond'] = samples['adj'][i].tolist()
            cur_mol['valid'] = samples['val'][i]
            save_data.append(cur_mol)
        with open(os.path.join('./sample_mols', f'{samples_save}.json'), 'w') as file:
            json_str = json.dumps(save_data, indent='\t')
            file.write(json_str)
            file.write('\n')
