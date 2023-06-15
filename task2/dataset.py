import os
import pickle as pkl
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

class MoleculeEvaluationDataset(Dataset):
    def __init__(self,dataset='train') -> None:
        super().__init__()
        datapath = os.path.join(f'../../Project for ML/MoleculeEvaluationData/{dataset}.pkl')
        with open(datapath,'rb') as f:
            file = pkl.load(f)
        self.data = list(file['packed_fp'])
        self.values = file['values'].double()
        for i,data in enumerate(self.data):
            self.data[i] = np.unpackbits(data)
        self.data = torch.from_numpy(np.array(self.data,dtype=float),)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index],self.values[index]

if __name__ == "__main__":
    dataset = MoleculeEvaluationDataset()
