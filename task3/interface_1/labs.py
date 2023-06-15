import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict

from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdchiral.main import rdchiralRunText
from sklearn.preprocessing import LabelEncoder
from rdchiral.template_extractor import extract_from_reaction
import sys

class OneStepModel():
    def __init__(self,args):
        args.device = 'cpu' if args.gpu < 0 else 'cuda:%i' % args.gpu
        self.device = torch.device(args.device)
        # Define the encoder
        with open('./interface_1/le.pkl', 'rb') as f:
            self.le = pickle.load(f)
        num_classes = len(self.le.classes_)
        self.model = MLP(args.input_size,args.hidden_size,num_classes)
        self.model.load_state_dict(torch.load('./interface_1/singe_model.pth'))
        self.model.to(self.device)


    def run(self,molecular, k):
        # Convert the fingerprint to a PyTorch tensor and add an extra dimension
        fp = smiles_to_fp(molecular)
        fp_tensor = torch.tensor(fp, dtype=torch.float).unsqueeze(0).to(self.device)

        # Get the model's raw predictions
        self.model.eval()
        with torch.no_grad():
            _, prob = self.model(fp_tensor)

        # Get the top k predictions
        topk_scores, topk_indices = torch.topk(prob, k)
        topk_scores = topk_scores.cpu().numpy().flatten()
        topk_indices = topk_indices.cpu().numpy().flatten()
        topk_templates = [label_to_template(self.le, idx) for idx in topk_indices]

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

    

def merge(reactant_d):
    ret = []
    for reactant, l in reactant_d.items():
        ss, ts = zip(*l)
        ret.append((reactant, sum(ss), list(ts)[0]))
    reactants, scores, templates = zip(*sorted(ret, key=lambda item: item[1], reverse=True))
    return list(reactants), list(scores), list(templates)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        probabilities = F.softmax(x, dim=1)
        return x, probabilities


def extract_template(reaction):
    try:
        reactants, products = reaction.split('>>')
        inputRec = {'_id': None, 'reactants': reactants, 'products': products}
        ans = extract_from_reaction(inputRec)
        if 'reaction_smarts' in ans.keys():
            template = ans['reaction_smarts']
            rec, pro = template.split('>>')
        else:
            template = None
    except Exception as e:
        print(f"Failed to extract template due to {e}")
        template = None
    return template


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


def label_to_template(le, label):
    # Convert the input to a 1D numpy array (expected by inverse_transform)
    label_arr = np.array([label])
    template_arr = le.inverse_transform(label_arr)

    # Extract the single element from the array and return
    return template_arr[0]


def evaluate_topk(model, dataloader, k):
    correct = 0
    total = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, probabilities = model(inputs)
        topk_prob, topk_indices = torch.topk(probabilities, k)
        total += labels.size(0)
        for i in range(len(labels)):
            if labels[i] in topk_indices[i]:
                correct += 1

    return 100 * correct / total



class RolloutPolicyNet(nn.Module):
    def __init__(self, n_rules, fp_dim=2048, dim=512,
                 dropout_rate=0.3):
        super(RolloutPolicyNet, self).__init__()
        self.fp_dim = fp_dim
        self.n_rules = n_rules
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(fp_dim,dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        # self.fc2 = nn.Linear(dim,dim)
        # self.bn2 = nn.BatchNorm1d(dim)
        # self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(dim,n_rules)

    def forward(self,x, y=None, loss_fn =nn.CrossEntropyLoss()):
        x = self.dropout1(F.elu(self.bn1(self.fc1(x))))
        # x = self.dropout1(F.elu(self.fc1(x)))
        # x = self.dropout2(F.elu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        if y is not None :
            return loss_fn(x, y)
        else :
            return x
        return x


def preprocess(X,fp_dim):

    # Compute fingerprint from mol to feature
    mol = Chem.MolFromSmiles(X)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(fp_dim),useChirality=True)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits())
    arr[onbits] = 1
    # arr = (arr - arr.mean())/(arr.std() + 0.000001)
    # arr = arr / fp_dim
    # X = fps_to_arr(X)
    return arr

class MLPModel(object):
    def __init__(self,state_path, template_path, device=-1, fp_dim=2048):
        super(MLPModel, self).__init__()
        self.fp_dim = fp_dim
        self.net, self.idx2rules = load_parallel_model(state_path,template_path, fp_dim)
        self.net.eval()
        self.device = device
        if device >= 0:
            self.net.to(device)

    def run(self, x, topk=10):
        arr = preprocess(x, self.fp_dim)
        arr = np.reshape(arr,[-1, arr.shape[0]])
        arr = torch.tensor(arr, dtype=torch.float32)
        if self.device >= 0:
            arr = arr.to(self.device)
        preds = self.net(arr)
        preds = F.softmax(preds,dim=1)
        if self.device >= 0:
            preds = preds.cpu()
        probs, idx = torch.topk(preds,k=topk)
        # probs = F.softmax(probs,dim=1)
        rule_k = [self.idx2rules[id] for id in idx[0].numpy().tolist()]
        reactants = []
        scores = []
        templates = []
        for i , rule in enumerate(rule_k):
            out1 = []
            try:
                out1 = rdchiralRunText(rule, x)
                # out1 = rdchiralRunText(rule, Chem.MolToSmiles(Chem.MolFromSmarts(x)))
                if len(out1) == 0: continue
                # if len(out1) > 1: print("more than two reactants."),print(out1)
                out1 = sorted(out1)
                for reactant in out1:
                    reactants.append(reactant)
                    scores.append(probs[0][i].item()/len(out1))
                    templates.append(rule)
            # out1 = rdchiralRunText(x, rule)
            except ValueError:
                pass
        if len(reactants) == 0: return None
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
        return {'reactants':reactants,
                'scores' : scores,
                'template' : templates}

def load_parallel_model(state_path, template_rule_path,fp_dim=2048):
    template_rules = {}
    with open(template_rule_path, 'r') as f:
        for i, l in tqdm(enumerate(f), desc='template rules'):
            rule= l.strip()
            template_rules[rule] = i
    idx2rule = {}
    for rule, idx in template_rules.items():
        idx2rule[idx] = rule
    rollout = RolloutPolicyNet(len(template_rules),fp_dim=fp_dim)
    checkpoint = torch.load(state_path,map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]
        new_state_dict[name] = v
    rollout.load_state_dict(new_state_dict)
    return rollout, idx2rule
