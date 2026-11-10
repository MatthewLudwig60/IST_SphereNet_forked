#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
import MDAnalysis as mda

anum = {'H':1, 'B':5, 'C':6, 'N':7,
        'O':8, 'F':9, 'P':15, 'S':16,
        'Cl':17, 'CL':17, 'Br':35, 'BR':35, 
        'I' : 53}
valid_elements = set(anum.keys())

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


if __name__ == '__main__':

    geom_dir = "/Users/hx24112/Documents/Bristol/PhD/Coding/tadf-htvs3-forked"
    df = pd.read_csv(os.path.join(os.getcwd(), "all_tddft_df.tsv"), sep="\t")
    files = [os.path.join(geom_dir, filename) for filename in df["Filename"]]
    geom_data = [get_geom(f) for f in files]
    Z, R = zip(*geom_data)
    N = np.asarray([len(i) for i in Z])
    Z = np.concatenate(Z)
    R = np.concatenate(R)
    Y = df["Singlet Triplet Splitting / eV"]
    mol = df["Name"]

    np.savez_compressed(
            "data_torch_heavy_only.npz",
            N=N,
            Z=Z,
            R=R,
            Y=Y,
            mol=mol,
            files=files
        )
