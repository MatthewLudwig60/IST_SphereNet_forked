#!/usr/bin/env python

import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from dig.threedgraph.method import SphereNet
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import MDAnalysis as mda


anum = {'H' : 1, 'C' : 6, 'N' : 7, 'O' : 8, 'F' : 9, 'Si' : 14, 
        'SI' : 14, 'P' : 15, 'S' : 16 ,'NA' : 11, 'Na' : 11, 'CL' : 17, 'Cl' : 17}

def get_sp2(u):
    '''
    Function to crop alkyl side chains in a universe, optionally writing
    a cropped trajectory.

    Parameters
    ----------
    u: object.
        MDAnalysis Universe to be cropped.

    Returns
    -------
    n: object.
        MDAnalysis Universe with cropped side chains.
    '''

    # Get bonds
    try:
        bds = u.bonds.to_indices()
    except:
        u.guess_bonds()
        bds = u.bonds.to_indices()

    # Get connectivity, use -1 as a placeholder for empty valence
    conn = np.ones((len(u.atoms), 4)) * -1
    for bond in bds:
        at1, at2 = bond
        for j in np.arange(conn[at1].shape[0]):
            if conn[at1,j] == -1:
                conn[at1,j] = at2
                break

        for j in np.arange(conn[at2].shape[0]):
            if conn[at2,j] == -1:
                conn[at2,j] = at1
                break

    # Get heavy atoms
    heavy = np.where(u.atoms.types != "H")[0]

    # Get sp3 atoms
    sat = np.where(np.all(conn > -1, axis=1))[0]

    allcheck = sat

    # Check all sp3 atoms
    keep = []
    delete = []
    for satat in allcheck:

        # check connectivity
        iconn = conn[satat]

        # filter H out from connected
        iconnheavy = iconn[np.in1d(iconn, heavy)]

        delete.append(satat)
        delete.extend(iconn[~np.in1d(iconn, heavy)])

    # Convert to int arrays
    keep = np.asarray(keep).astype(int)
    delete = np.asarray(delete).astype(int)

    # Get non sp3 atoms
    unsat = ~np.all(conn > -1, axis=1)

    # Set which saturated atoms to keep or delete
    unsat[keep] = True
    unsat[delete] = False
    tokeep = np.where(unsat)[0]

    return tokeep

def predict_from_xyz_files(filenames, chk="train_30.pt", batch_size=1):
    """
    args: 
    
    filename (str): path to xyz file
    checkpoint (str): path to checkpoint file

    Description:

    Makes prediction from XYZ file with dimer coordinates
    """

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

    all_data = []
    for filename in filenames:
        u = mda.Universe(filename, guess_bonds=True)
        Z = np.asarray(list(map(anum.get, u.atoms.types)))
        if None in Z:
            raise ValueError(f"Unsupported element in {filename}")
        sp2idxs = get_sp2(u)
        Z = Z[sp2idxs]
        X = u.atoms.positions[sp2idxs]
        idxs = np.where(Z != 1)[0]
        R_i = torch.tensor(X[idxs], dtype=torch.float32)
        z_i = torch.tensor(Z[idxs], dtype=torch.int64)
        data = Data(pos=R_i, z=z_i)
        all_data.append(data)
    batch = DataLoader(all_data, batch_size=batch_size)

    return pred_data(model, batch)


def pred_data(model, batch):

    yhat = []
    model.eval()
    for data in tqdm(batch):
        data.to(gpu)
        try:
            with torch.no_grad():
                pred = model(data)
            yhat.append(pred.cpu().detach().numpy())
        except Exception as exc:
            raise RuntimeError(f"Prediction failed for batch") from exc

    yhat = np.concatenate(yhat).reshape(-1)

    return yhat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict singlet-triplet splittings from XYZ geometries."
    )
    parser.add_argument(
        "input_df",
        type=Path,
        help="Input CSV or TSV containing a Filename column.",
    )
    parser.add_argument(
        "xyz_parent_dir",
        type=Path,
        help="Directory containing the paths listed in Filename.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        default=Path(__file__).with_name("train_30.pt"),
        help="SphereNet checkpoint path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("IST_spherenet_predictions.tsv"),
        help="Output TSV path.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
        help="Inference device; defaults to CUDA when available.",
    )
    args = parser.parse_args()

    if not args.input_df.is_file():
        parser.error(f"Input file does not exist: {args.input_df}")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available.")

    gpu = torch.device(args.device)

    sep = "\t" if args.input_df.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(args.input_df, sep=sep)

    if "Filename" not in df.columns:
        parser.error("Input file must contain a 'Filename' column.")

    filenames = [
        str(args.xyz_parent_dir / filename)
        for filename in df["Filename"]
    ]

    df["ML Singlet Triplet Splitting / eV"] = predict_from_xyz_files(
        filenames,
        chk=str(args.checkpoint),
        batch_size=1,
    )
    df.to_csv(args.output, index=False, sep="\t")
