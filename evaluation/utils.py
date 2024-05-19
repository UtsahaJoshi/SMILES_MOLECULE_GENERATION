
import os
import pkgutil
import tempfile
from rdkit import Chem
from multiprocessing import Pool
import fcd
import numpy as np


def loadmodel():
    chemnet_model_filename = 'ChemNet_v0.13_pretrained.pt'
    model_bytes = pkgutil.get_data('fcd', chemnet_model_filename)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'chemnet.pt')

        with open(model_path, 'wb') as f:
            f.write(model_bytes)

        print(f'Saved ChemNet model to \'{model_path}\'')

        return fcd.load_ref_model(model_path)


def getstats(smiles, model):
    predictions = fcd.get_predictions(model, smiles)
    mean = predictions.mean(0)
    cov = np.cov(predictions.T)
    return mean, cov

def _cansmi(smi):
    """Try except is needed in case rdkit throws an error"""
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        can_smi = Chem.MolToSmiles(mol)
    except:
        can_smi = None
    return can_smi


def canonicalize_smiles(smiles, njobs=8):
    r"""calculates canonical smiles
    Arguments:
        smiles (list): List of smiles
        njobs (int): How many workers to use

    Returns:
        canonical_smiles: A list of canonical smiles. None if invalid smiles.
    """

    with Pool(njobs) as pool:
        # pairs of mols and canonical smiles
        canonical_smiles = pool.map(_cansmi, smiles)

    return canonical_smiles