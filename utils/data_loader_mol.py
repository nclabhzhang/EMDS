import os
from time import time
import numpy as np
import networkx as nx

import torch
from torch.utils.data import DataLoader, Dataset


def load_mol_3d(filepath):
    print(f'Loading file {filepath}')
    if not os.path.exists(filepath):
        raise ValueError(f'Invalid filepath {filepath} for dataset')
    np.load.__defaults__ = (None, True, True, 'ASCII')
    load_data = np.load(filepath)
    np.load.__defaults__ = (None, False, True, 'ASCII')
    result = []
    i = 0
    while True:
        key = f'arr_{i}'
        if key in load_data.keys():
            result.append(load_data[key])
            i += 1
        else:
            break
    return list(map(lambda x, y, z: (x, y, z), result[0], result[1], result[2]))


class MolDataset(Dataset):
    def __init__(self, mols, transform):
        self.mols = mols
        self.transform = transform

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        return self.transform(self.mols[idx])


def get_3d_transform_fn(dataset):
    def transform(data):
        x, pos, adj = data
        x = torch.tensor(x).to(torch.float32)
        pos = torch.tensor(pos).to(torch.float32)
        adj = torch.tensor(adj).to(torch.float32)
        return x, pos, adj
    return transform

def dataloader_3d(config, get_graph_list=False):
    start_time = time()

    mols = load_mol_3d(os.path.join(config.data.dir, f'{config.data.data.lower()}_kekulized.npz'))

    split_idxs = np.load(os.path.join(config.data.dir, 'split.npz'))
    split_dict = {
        'train': split_idxs['train_idx'].tolist(),
        'valid': split_idxs['val_idx'].tolist(),
        'test': split_idxs['test_idx'].tolist()
    }

    train_mols = [mols[i] for i in split_dict['train']]
    valid_mols = [mols[i] for i in split_dict['valid']]
    test_mols = [mols[i] for i in split_dict['test']]

    print(f'Number of training mols: {len(split_dict["train"])} | Number of test mols: {len(split_dict["test"])}')

    train_dataset = MolDataset(train_mols, get_3d_transform_fn(config.data.data))
    test_dataset = MolDataset(test_mols, get_3d_transform_fn(config.data.data))

    if get_graph_list:
        train_mols_nx = [nx.from_numpy_matrix(np.array(adj)) for x, adj in train_dataset]
        test_mols_nx = [nx.from_numpy_matrix(np.array(adj)) for x, adj in test_dataset]
        return train_mols_nx, test_mols_nx

    train_dataloader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.data.batch_size, shuffle=True)

    print(f'{time() - start_time:.2f} sec elapsed for data loading')
    return train_dataloader, test_dataloader
