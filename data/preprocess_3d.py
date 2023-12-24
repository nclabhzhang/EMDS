import os
import numpy as np
import argparse
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from tqdm import tqdm

# can use the url to download dataset
GDB9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
QM9_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"

def load_mol(filepath):
    print(f'Loading file {filepath}')
    if not os.path.exists(filepath):
        raise ValueError(f'Invalid filepath {filepath} for dataset')

    load_data = np.load(filepath)
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


class QM93D(Dataset):
    def __init__(self, mols):
        super(QM93D, self).__init__()
        self.atom_type_list, self.position_list, self.con_mat_list = mols

    def __getitem__(self, idx):
        return self.atom_type_list[idx], self.position_list[idx], self.con_mat_list[idx]

    def __len__(self):
        return len(self.atom_type_list)


def process(sdf_path, data_name):
    mols = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    atom_type_list, position_list, con_mat_list = [], [], []

    for idx in tqdm(range(len(mols))):
        mol = mols[idx]
        num_atoms = mol.GetNumAtoms()
        position = mols.GetItemText(idx).split('\n')[4:4 + num_atoms]
        position = np.array([[float(x) for x in line.split()[:3]] for line in position], dtype=np.float32)
        atom_type = np.array([atomic_num_to_type[atom.GetAtomicNum()] for atom in mol.GetAtoms()])

        con_mat = np.zeros([num_atoms, num_atoms], dtype=int)
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = bond_to_type[bond.GetBondType()]
            con_mat[start, end] = bond_type
            con_mat[end, start] = bond_type

        x_ = np.zeros((MAX_NUM, 6))
        atom_type_ = np.pad(atom_type, [0, MAX_NUM - len(atom_type)], constant_values=(0, 5))
        x_[np.arange(29), atom_type_] = 1
        atom_type_al = x_[:, :-1]

        position_al = np.pad(position, [(0, MAX_NUM-len(atom_type)), (0, 0)])

        con_mat_al = np.pad(con_mat, [(0, MAX_NUM-len(atom_type)), (0, MAX_NUM-len(atom_type))])

        atom_type_list.append(atom_type_al)
        position_list.append(position_al)
        con_mat_list.append(con_mat_al)

    np.savez(f'data/{data_name.lower()}_kekulized.npz', atom_type_list, position_list, con_mat_list)


if __name__ == '__main__':
    atomic_num_to_type = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
    bond_to_type = {BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3}
    MAX_NUM = 29

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='QM93D')
    args = parser.parse_args()

    dataset_file = 'gdb9.sdf'
    process(dataset_file, args.dataset)
