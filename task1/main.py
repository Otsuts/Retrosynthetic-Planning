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



class MyDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        features = []
        # labels = []
        productss = []
        gts = []
        templates = []
        # tmp = set()
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                id = row[0]
                reaction = row[2]
                reactants, products = reaction.split('>>')
                productss.append(products)
                gts.append(reactants)
                inputRec = {'_id': id, 'reactants': reactants, 'products': products}
                # products是我们的输入要存起来
                ans = extract_from_reaction(inputRec)
                # ans['reaction_smarts']是我们的模板
                # print(ans['reaction_smarts'])
                # tmp.add(ans['reaction_smarts'])
                templates.append(ans['reaction_smarts'])
                # labels.append(len(tmp))
                mol = Chem.MolFromSmiles(products)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                onbits = list(fp.GetOnBits())
                arr = np.zeros(fp.GetNumBits())
                arr[onbits] = 1
                # print(arr.shape)
                # print(len(tmp))
                features.append(arr)
        # self.label_num = len(tmp)
        self.features = features
        # self.labels = torch.tensor(labels,dtype=torch.int)
        self.templates = templates
        self.products = productss
        self.gts = gts
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, item):
        return self.features[item], self.templates[item], self.products[item], self.gts[item]
    

class SP(torch.nn.Module):
    def __init__(self):
        super(SP, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.Sigmoid(),

            # torch.nn.Dropout(0.2),
            torch.nn.Linear(512,10263),
            # torch.nn.Softmax()
        )
        
    def forward(self, x):
        return self.mlp(x)
    
def train(topk,dict,new_dict, device, model, train_loader, val_loader,test_loader, train_size, val_size,test_size, num_epochs, lr, weight_decay):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, lr/1000)
    # loss_value = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    best = -math.inf
    best_epoch = 1
    return_list = []

    for ep in range(1, num_epochs):
        acc_num = 0
        # cnt = 0
        # loss_record = []
        model.train()
        pbar = tqdm(train_loader)
        for i, batch in enumerate(pbar):
            # print(batch)
            feature = torch.tensor(batch[0]).to(device)
            feature = feature.to(torch.float)
            # print("1")
            template = batch[1] #template
            gt = batch[3] #reactants
            products = batch[2] #products
            y = model(feature)

            with torch.no_grad():
                result = torch.argmax(y, dim=-1)
                # print(len(batch))
                for j in range(len(batch[0])):
                    predict = new_dict[result[j].item()]

                    if predict == template[j]:
                        acc_num+=1

                    
            y = y.to(torch.float)
            # print(gtt)
            gt_list = []
            for k in range(len(batch[0])):
                gta = dict[template[k]]
                # print(gta)
                tt = torch.zeros(10263)
                tt[gta]=1
                # print(tt)
                gt_list.append(tt)
                # cnt+=1
            groundTruth = torch.stack(gt_list).to(device)
            # groundTruth = torch.from_numpy(np.array(gt_list))
            # print()
            # print(groundTruth.shape)
            
            # print(y.shape)
            # groundTruth = torch.tensor(dict[template]).reshape(-1).to(device)
            # groundTruth = groundTruth.to(torch.long)
            # p = torch.nn.functional.softmax(y, dim=-1)
            # print(p.shape)
            loss = loss_fn(input=y,target= groundTruth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"loss: {loss:.5f}")
        lr_scheduler.step()
        # mean_train_loss = sum(loss_record)/train_size
        train_acc = acc_num/ train_size
        print('Epoch{}: training accuracy:{:.4f}, training loss:{:.4f}'.format(ep, train_acc, loss))
        # cnt = 0
        model.eval()
        acc_num = 0
        pbar = tqdm(val_loader)
        for i,batch in enumerate(pbar):
            feature = torch.tensor(batch[0]).to(device)
            feature = feature.to(torch.float)
            # print("1")
            template = batch[1] #template
            gt = batch[3] #reactants
            products = batch[2] #products
            y = model(feature)
            # print(y)
            # print(new_dict)
            # gtt = []
            with torch.no_grad():
                result = torch.argmax(y, dim=-1)
                # print(len(batch))
                for j in range(len(batch[0])):
                    predict = new_dict[result[j].item()]

                    if predict == template[j]:
                        acc_num+=1
                    # print(acc_num)
                    # 这个和gt比
                    
                y = y.to(torch.float)
            # print(gtt)
                gt_list = []
                 
                for k in range(len(batch[0])):
                  
                # print(gta)
                    tt = torch.zeros(10263)
                    if template[k] in dict.keys():
                        gta = dict[template[k]]
                        tt[gta]=1
                    # cnt+=1     
                # print(tt)
                    gt_list.append(tt)
                groundTruth = torch.stack(gt_list).to(device)
            # groundTruth = torch.from_numpy(np.array(gt_list))
            # print()
            # print(groundTruth.shape)
            
            # print(y.reshape(-1,10263).shape)
            # groundTruth = torch.tensor(dict[template]).reshape(-1).to(device)
            # groundTruth = groundTruth.to(torch.long)
                loss = loss_fn(input=y.reshape(-1,10263),target= groundTruth.reshape(-1, 10263))


                # pbar.set_description(f"loss: {loss:.5f}")
        # lr_scheduler.step()
        # mean_train_loss = sum(loss_record)/train_size
        valid_acc = acc_num/val_size
        print('Epoch{}: valid accuracy:{:.4f}, valid loss:{:.4f}'.format(ep, valid_acc, loss))
        # print(cnt)
        
        
        # 测试环节
        acc_num = 0
        tt_acc_num = 0
        pbar = tqdm(test_loader)
        for i,batch in enumerate(pbar):
            feature = torch.tensor(batch[0]).to(device)
            feature = feature.to(torch.float)
            template = batch[1] #template
            gt = batch[3] #reactants
            products = batch[2] #products
            y = model(feature)

            with torch.no_grad():
                result = torch.argmax(y, dim=-1)
                for j in range(len(batch[0])):
                    predict = new_dict[result[j].item()]
                    if predict == template[j]:
                        acc_num+=1
                # top10
                values, indices = y.topk(topk, dim=1, largest=True)
                # print(indices)
                for j in range(len(batch[0])):
                    a = list(indices[j])
                    # print(a)
                    for it in range(len(a)):
                        predict = new_dict[a[it].item()]
                        if predict == template[j]:
                            tt_acc_num+=1
                            break
                        
                # break
                y = y.to(torch.float)

                gt_list = []    
                for k in range(len(batch[0])):
                    tt = torch.zeros(10263)
                    if template[k] in dict.keys():
                        gta = dict[template[k]]
                        tt[gta]=1
                    gt_list.append(tt)
                groundTruth = torch.stack(gt_list).to(device)
                loss = loss_fn(input=y.reshape(-1,10263),target= groundTruth.reshape(-1, 10263))
        test_acc = acc_num/test_size
        
        tt_acc = tt_acc_num/test_size
        return_list.append(tt_acc)
        if tt_acc > best:
            best = tt_acc
            best_epoch = ep
            if not os.path.exists('saved_models'):
                os.mkdir('saved_models')
            torch.save(model, './saved_models/best_model.pth')
        print('Epoch{}: test accuracy:{:.4f},top 10 test acc:{:.4f}, test loss:{:.4f}'.format(ep, test_acc,tt_acc, loss))
        # print(cnt)
        
    return return_list
                    
                    
                    
                    
def get_printer(chemistry):
    mol = Chem.MolFromSmiles(chemistry)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=bool)
    arr[onbits] = 1
    return arr          
                
if __name__ == '__main__':
    # train_dataset = MyDataset(path='data/1/raw_train.csv')
    device = 'cuda'
    device = torch.device(device)  
    # print(train_dataset.__len__())
    # valid_dataset = MyDataset(path='data/1/raw_val.csv')
    # test_dataset = MyDataset(path='data/1/raw_test.csv')
    
    # with open('train_data.pkl', 'wb') as f:
    #     dill.dump(train_dataset, f)
    # with open('valid_data.pkl', 'wb') as f:
    #     dill.dump(valid_dataset, f)
    # with open('test_data.pkl', 'wb') as f:
    #     dill.dump(test_dataset, f)
    with open('train_data.pkl', 'rb') as f:
        train_dataset = dill.load(f)
    with open('valid_data.pkl', 'rb') as f:
        valid_dataset = dill.load(f)
    with open('test_data.pkl', 'rb') as f:
        test_dataset = dill.load(f)
    
    # 建立一个字典
    # print(train_dataset)
    templateSet = set()
    newDict = {}
    tempDict = {}
    for i in range(train_dataset.__len__()):
        product, template = train_dataset.features[i], train_dataset.templates[i]
        t = len(templateSet)
        templateSet.add(template)
        if t!=len(templateSet):
        
        # label.append(len(templateSet))
            tempDict[template] = len(templateSet)-1
            newDict[len(templateSet)-1]=template
            
    with open('index2template.pkl', 'wb') as f:
        pickle.dump(newDict, f)
    with open('template2index.pkl', 'wb') as f:
        pickle.dump(tempDict, f)

    
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=512)
    test_loader = DataLoader(test_dataset, batch_size=512)
    import matplotlib.pyplot as plt
    for k in 1, 3, 5, 10:
        model = SP().to(device)
        draw_list = train(k,tempDict,newDict,device,model,train_loader,val_loader,test_loader, train_dataset.__len__(),valid_dataset.__len__(),test_dataset.__len__(),100,1e-3,1e-6)
        ep_list = list(range(len(draw_list)))
        plt.plot(ep_list, draw_list, label=str(k))
        

    plt.title('Task1: Testing accuracy@Top-K')
    plt.xlabel('epoch')
    plt.ylabel('Test acc')
    plt.legend()
    plt.savefig('K.png')   

# with open('data/1/raw_train.csv') as f:
#     reader = csv.reader(f)
#     header = next(reader)
#     for row in reader:
#         id = row[0]
#         reaction = row[2]
#         reactants, products = reaction.split('>>')d

#         inputRec = {'_id': id, 'reactants': reactants, 'products': products}
#         ans = extract_from_reaction(inputRec)
#         # print(ans)
#         print(products)
#         print(ans['reaction_smarts'])
#         t = ans['reaction_smarts']
#         time.sleep(1)

#         mol = Chem.MolFromSmiles(products)
#         fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
#         onbits = list(fp.GetOnBits())
#         arr = np.zeros(fp.GetNumBits())
#         arr[onbits] = 1
#         # print(arr.shape)
#         out = rdchiralRunText(t, products)
#         print(out)
