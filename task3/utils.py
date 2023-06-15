import pickle as pkl
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from interface_1.labs import OneStepModel
from interface_2.labs import TrialModel
from retro import molstar
import torch

def prepare_starting_molecules(filename):
    with open(filename, 'rb')as f:
        starting_mols = pkl.load(f)
    return set(starting_mols)

def get_task1_model(args):
    model = OneStepModel(args)
    return model

def get_task2_model(args):
    model = TrialModel()
    model.to(args.device)
    model.load_state_dict(torch.load(args.tsk2_model_path))
    return model

def smiles_to_fp(s, fp_dim=2048, pack=False):
    mol = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool)
    arr[onbits] = 1

    if pack:
        arr = np.packbits(arr)

    return arr

def prepare_molstar_planner(one_step, value_fn, starting_mols, expansion_topk,
                            iterations, viz=False, viz_dir=None):
    expansion_handle = lambda x: one_step.run(x, expansion_topk)

    plan_handle = lambda x, y=0: molstar(
        target_mol=x,
        target_mol_id=y,
        starting_mols=starting_mols,
        expand_fn=expansion_handle,
        value_fn=value_fn,
        iterations=iterations,
        viz=viz,
        viz_dir=viz_dir
    )
    return plan_handle

