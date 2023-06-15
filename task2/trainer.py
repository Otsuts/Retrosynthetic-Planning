import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import TrialModel
from dataset import MoleculeEvaluationDataset
from tqdm import tqdm


class Trainer():
    def __init__(self, args):
        self.args = args
        self.model = TrialModel()
        args.device = 'cpu' if args.device < 0 else 'cuda:%i' % args.device
        self.device = torch.device(args.device)
        self.model.to(self.device)
        print(f'device: {self.device}')
        self.train_data = MoleculeEvaluationDataset('train')
        self.test_data = MoleculeEvaluationDataset('test')
        self.train_loader = DataLoader(
            self.train_data, batch_size=2048, shuffle=True)
        self.test_loader = DataLoader(
            self.test_data, batch_size=2048, shuffle=False)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.loss_func = F.mse_loss
        self.num_epochs = args.num_epochs
        self.test_eval = args.eval_iter

    def train_one_epoch(self,):
        self.model.train()
        for data, value in tqdm(self.train_data):
            data,value = data.to(self.device),value.to(self.device)
            pred = self.model(data)
            
            loss = self.loss_func(pred, value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self, dataset='train'):
        dataloader = self.train_loader if dataset == 'train' else self.test_loader
        loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for data, value in dataloader:
                data,value = data.to(self.device),value.to(self.device)
                pred = self.model(data)
                loss += self.loss_func(pred, value)
        print(f'{dataset} loss: {loss}')
        return loss

    def train(self):
        best_loss = float('inf')
        for epoch in range(self.num_epochs):
            print(f'=========== Epoch {epoch} =============')
            self.train_one_epoch()
            self.evaluate('train')
            if not (epoch+1) % self.test_eval:
                loss = self.evaluate('test')
                if loss < best_loss:
                    best_loss = loss
                    torch.save(self.model.state_dict(),self.args.save_dir+f'model_LR{self.args.lr}.pth')
                
