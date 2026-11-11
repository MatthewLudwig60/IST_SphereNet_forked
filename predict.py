#!/usr/bin/env python

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from dig.threedgraph.method import SphereNet
from dig.threedgraph.evaluation import ThreeDEvaluator
from dig.threedgraph.method import run

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import MDAnalysis as mda

from datasets import *


anum = {'H':1, 'B':5, 'C':6, 'N':7,
        'O':8, 'F':9, 'P':15, 'S':16,
        'Cl':17, 'CL':17, 'Br':35, 'BR':35, 
        'I' : 53}
valid_elements = set(anum.keys())


def flatten(lst):
    '''
    Recursive function to flatten a nested list.

    Parameters
    ----------
    lst: list.
        Nested list to be flattened.

    Returns
    -------                                             
    flattened: list.
        Flattened list.
    '''

    flattened = sum( ([x] if not isinstance(x, list)
                     else flatten(x) for x in lst), [] )

    return flattened 

def get_geom(fname, remove_hydrogens=True):

    u = mda.Universe(fname)
    elements = set(u.atoms.types)
    if not elements.issubset(valid_elements):
        raise ValueError(f"Invalid elements found in {fname}: {elements - valid_elements}")
    Z = np.asarray(list(map(anum.get, u.atoms.types)))
    X = u.atoms.positions
    if remove_hydrogens:
        idxs = np.where(Z != 1)[0]
        Z = Z[idxs]
        X = X[idxs]
    return Z, X

def predict(batch, checkpoint):

    model = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4,
                      hidden_channels=128, out_channels=1, int_emb_size=64,
                      basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8,
                      out_emb_channels=256, num_spherical=3, num_radial=6, envelope_exponent=5,
                      num_before_skip=1, num_after_skip=2, num_output_layers=3)

    model.load_state_dict(
            torch.load(checkpoint, map_location=torch.device('cuda:0'))["model_state_dict"]
        )
    model.eval()

    results = []
    for data in batch:
        prediction = model(data)
        results.append(prediction.detach().numpy())

    return results


def predict_from_file(filename, checkpoint):
    """
    args: 
    
    filename (str): path to xyz file
    checkpoint (str): path to checkpoint file

    Description:

    Makes prediction from XYZ file with dimer coordinates
    """

    Z, X = get_geom(filename)
    R_i = torch.tensor(X, dtype=torch.float32)
    z_i = torch.tensor(Z, dtype=torch.int64)
    data = Data(pos=R_i, z=z_i)
    batch = DataLoader([data], batch_size=1)

    return predict(batch,checkpoint=checkpoint)


def pred_data(model, dataset):

    # Get predictions
    batch = DataLoader(dataset, batch_size=8)

    yhat = []
    mols = []
    for data in tqdm(batch):
        mols.append(data.mol)
        data.to(gpu)
        try:
            pred = model(data)
            yhat.append(pred.cpu().detach().numpy())
        except:
            yhat.append(np.zeros((8, 1)))

    yhat = np.concatenate(yhat).reshape(-1)

    df = pd.DataFrame({
                "name" : flatten(mols),
                "ST_ML  / meV" : yhat * 1000,
           })

    return df


if __name__ == '__main__':

    dataset = Geoms()

    split_idx = torch.load("splits.pt")
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

    chk = f"train_25.pt"
    gpu = torch.device('cuda:0')

    # Define model
    model = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4,
                      hidden_channels=128, out_channels=1, int_emb_size=64,
                      basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8,
                      out_emb_channels=256, num_spherical=3, num_radial=6, envelope_exponent=5,
                      num_before_skip=1, num_after_skip=2, num_output_layers=3)                 

    model.load_state_dict(
            torch.load(chk, map_location=gpu)["model_state_dict"]
        )
    model.to(gpu)

    df = pred_data(model, test_dataset) 
    df.to_csv(f"predictions.tsv", index=False, sep='\t')
