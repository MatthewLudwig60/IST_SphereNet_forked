#!/usr/bin/env python

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.utils import shuffle
import yaml

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset


class Geoms(InMemoryDataset):
    def __init__(self, root='geometries/', transform=None, pre_transform=None, pre_filter=None):

        self.folder = os.path.join(root, 'data')
        super(Geoms, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data_torch_heavy_only.npz'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        
        data = np.load(os.path.join(self.raw_dir, self.raw_file_names), allow_pickle=True)
        N = data['N']
        R = data['R']
        Z = data['Z']
        mol = data['mol']
        split = np.cumsum(N)
        R_dim = np.split(R, split)
        Z_dim = np.split(Z, split)
        target = {}
        target["Y"] = np.expand_dims(data["Y"],axis=-1)

        data_list = []
        for i in tqdm(range(len(N))):
            R_i = torch.tensor(R_dim[i],dtype=torch.float32)
            z_i = torch.tensor(Z_dim[i],dtype=torch.int64)
            y_i = torch.tensor(target["Y"][i],dtype=torch.float32)
            mol_i = mol[i]
            data = Data(
                    pos=R_i,
                    z=z_i,
                    y=y_i,
                    mol=mol_i,
                    shift="0",
                )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [ data for data in data_list if self.pre_filter(data) ]
        if self.pre_transform is not None:
            data_list = [ self.pre_transform(data) for data in data_list ]
        
        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def random_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict

    def load_idx_split_from_yaml(self, yaml_file, valid_split, seed):
        with open(yaml_file, 'r') as f:
            split_dict = yaml.load(f, Loader=yaml.Loader)

        parent_mol_type = {}
        for mol_type, parent_list in split_dict['Parent Molecules'].items():
            for parent in parent_list:
                parent_mol_type[parent] = mol_type
        
        train_mols, test_mols = [], []
        for i, mol in enumerate(self._data.mol):
            parent = mol.split('_')[0]
            if mol in split_dict['All Molecules Testing'][parent_mol_type[parent]][parent]:
                test_mols.append(i)
            else:
                train_mols.append(i)
        train_ids = shuffle(range(len(train_mols)), random_state=seed)
        train_size = int(round(len(train_mols) * (1 - valid_split)))
        train_idx = torch.tensor([train_mols[i] for i in train_ids[:train_size]])
        val_idx = torch.tensor([train_mols[i] for i in train_ids[train_size:]])
        test_idx = torch.tensor(test_mols)
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict

if __name__ == '__main__':

    dataset = Geoms()

    try:
        split_idx = torch.load("splits.pt")
    except:
        print('except')
        # split_idx = dataset.get_idx_split(len(dataset._data.y), train_size=41400, valid_size=13800, seed=42)
        split_idx = dataset.load_idx_split_from_yaml('molecule_list.yaml', valid_split=0.2, seed=42)
        torch.save(split_idx, "splits.pt")

    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
