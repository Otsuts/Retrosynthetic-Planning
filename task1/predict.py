import csv
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import dill
sys.path.append('..')
from rdchiral.template_extractor import extract_from_reaction
from rdchiral.main import rdchiralRunText
import time
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import DataLoader
import torch.nn
import math
from tqdm import tqdm
import numpy as np
import os
import pickle
from main import SP
from collections import defaultdict
def merge(reactant_d):
    ret = []
    for reactant, l in reactant_d.items():
        ss, ts = zip(*l)
        ret.append((reactant, sum(ss), list(ts)[0]))
    reactants, scores, templates = zip(*sorted(ret, key=lambda item: item[1], reverse=True))
    return list(reactants), list(scores), list(templates)

def smiles_to_fp(products):
    try:
        mol = Chem.MolFromSmiles(products)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        onbits = list(fp.GetOnBits())
        arr = np.zeros(fp.GetNumBits(), dtype=bool)
        arr[onbits] = 1
    except Exception as e:
        print(f"Failed to convert to fingerprint due to {e}")
        arr = None
    return arr

def recover_reaction(template, products):
    try:
        out = rdchiralRunText(template, products)
    except Exception as e:
        print(f"Failed to recover reaction due to {e}")
        out = None
    return out

def predict_topk(molecular, k):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('index2template.pkl', 'rb') as f:
        index2template = pickle.load(f)
    with open('template2index.pkl', 'rb') as f:
        template2index = pickle.load(f)
    
    model = SP()
    model = torch.load('saved_models/best_model.pth')
    fp = smiles_to_fp(molecular)
    fp_tensor = torch.tensor(fp, dtype=torch.float).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        y = model(fp_tensor)
    # print(y)
    y = torch.nn.functional.softmax(y)
    topk_scores, topk_indices = torch.topk(y, k)
    topk_scores = topk_scores.cpu().numpy().flatten()
    topk_indices = topk_indices.cpu().numpy().flatten()
    topk_templates = [index2template[idx] for idx in topk_indices]

    reactants = []
    scores = []
    templates = []


    for i, template in enumerate(topk_templates):
        out = recover_reaction(template, molecular)
        if out is not None:
            for reactant in out:
                reactants.append(reactant)
                scores.append(topk_scores[i] / len(out))
                templates.append(template)

    if len(reactants) == 0:
        return None

    reactants_d = defaultdict(list)
    for r, s, t in zip(reactants, scores, templates):
        if '.' in r:
            str_list = sorted(r.strip().split('.'))
            reactants_d['.'.join(str_list)].append((s, t))
        else:
            reactants_d[r].append((s, t))

    reactants, scores, templates = merge(reactants_d)
    total = sum(scores)
    scores = [s / total for s in scores]

    return {
        'reactants': reactants,
        'scores': scores,
        'templates': templates
    }
    
    # print(y)
    # print(y.shape)
    
    
if __name__ == "__main__":
    molecular_str = "[C:1](=[O:2])([C:3]([F:4])([F:5])[F:6])[NH:7][c:8]1[cH:9][cH:10][c:11]([O:12][c:13]2[cH:14][cH:15][n:16][c:17]3[nH:18][cH:19][cH:20][c:21]23)[c:22]([F:23])[cH:24]1"
    
    print(predict_topk(molecular_str, 10))