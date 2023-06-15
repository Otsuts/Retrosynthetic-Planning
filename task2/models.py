import torch.nn as nn

class TrialModel(nn.Module):
    def __init__(self,min = 0,max = 46):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048,256,dtype=float),
            # nn.BatchNorm1d(),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(256,1,dtype=float),
            nn.Sigmoid()
        )
        self.min = min
        self.max = max
    
    def forward(self,X):
        X = self.fc(X)
        return self.min + X*(self.max-self.min)
    


