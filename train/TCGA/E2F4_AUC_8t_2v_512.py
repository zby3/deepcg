import sys
sys.path.remove('/opt/apps/intel18/impi18_0/python2/2.7.16/lib/python2.7/site-packages')
import os
import pandas as pd
from glob import glob
from PIL import Image
import yaml
import numpy as np
from tqdm import tqdm
import torch
from dgl.data.utils import load_graphs
import torch.nn as nn
from histocartography.ml import CellGraphModel
import argparse
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import copy

parser = argparse.ArgumentParser(description='E2F4')
parser.add_argument('--lr', type=float,help='learning rate')
parser.add_argument('--l2', type=float,help='l2 regularization')
parser.add_argument('--c', type=float,help='checkpoint')
parser.add_argument('--mpath', type=str,help='model path')
parser.add_argument('--e', type=int,help='epochs')
args = parser.parse_args()

IS_CUDA = torch.cuda.is_available()
device = 'cuda:0' if IS_CUDA else 'cpu'
NODE_DIM = 512


config_fname="/work/07034/byz/maverick2/myapps/patho-quant-explainer/core/config/E2F41.yml"

with open(config_fname, 'r') as file:
    config = yaml.safe_load(file)
model = CellGraphModel(
        gnn_params=config['gnn_params'],
        classification_params=config['classification_params'],
        node_dim=NODE_DIM,
        num_classes=2,
        pretrained=False
    ).to(device)
if args.c==1:
    model.load_state_dict(torch.load(args.mpath))

###-----------------------load graphs
import random
all1="/work/07034/byz/maverick2/GNN/LUAD_CG/*.bin"
all2="/work/07034/byz/maverick2/GNN/LUBD_CG/*.bin"
all3="/work/07034/byz/maverick2/GNN/LUCD_CG/*.bin"
all1=glob(all1)
all2=glob(all2)
all3=glob(all3)
all=all1+all2+all3
myinf1="/work/07034/byz/maverick2/GNN/TCGA_LUAD_E2F4_iRAS.txt"
info=pd.read_csv(myinf1,sep="\t")
sam1=np.where(info.loc[:,"all.intersection.ES"]>0)
sam2=np.where(info.loc[:,"all.intersection.ES"]<=0)
info1=info.iloc[sam1]
info2=info.iloc[sam2]
POS=list(info1.index)
NEG=list(info2.index)
allid=POS+NEG
def match_data(path,poslist,neglist):
    patient_id = []
    all_path = path
    for i in range(len(all_path)):
        patient_id.append(all_path[i].split("/")[7][0:12])
    t1 = pd.DataFrame({'ID': patient_id, 'path': all_path})
    pos_se = np.where(t1.loc[:, "ID"].isin(poslist))
    neg_se = np.where(t1.loc[:, "ID"].isin(neglist))
    nposid=t1.iloc[pos_se]["ID"].tolist()
    nnegid=t1.iloc[neg_se]["ID"].tolist()
    npos=len(nposid)
    nneg=len(nnegid)
    nposid=np.unique(nposid)
    nnegid = np.unique(nnegid)
    nposid=len(nposid)
    nnegid=len(nnegid)
    print("tiles level:","all:",len(all_path),"pos:",npos,"neg",nneg)
    print("patient level:","pos",nposid,"neg",nnegid,"all:",len(np.unique(patient_id)))
    return None
match_data(all,POS,NEG)

def split_train_val_test(path,ratio,idlist):
    patient_id = []
    all_path=path
    for i in range(len(all_path)):
        patient_id.append(all_path[i].split("/")[7][0:12])
    t1 = pd.DataFrame({'ID': patient_id, 'path': all_path})
    se=np.where(t1.loc[:, "ID"].isin(idlist))
    t1=t1.iloc[se]
    ID=t1["ID"].tolist()
    ID=np.unique(ID)
    random.shuffle(ID)
    train_ID=ID[0:round(ratio[0] * len(ID))]
    val_ID=ID[round(ratio[0] * len(ID)):round((ratio[0]+ratio[1]) * len(ID))]
    test_ID=ID[round((ratio[0]+ratio[1]) * len(ID)):]
    train_se = np.where(t1.loc[:, "ID"].isin(train_ID))
    val_se = np.where(t1.loc[:, "ID"].isin(val_ID))
    test_se = np.where(t1.loc[:, "ID"].isin(test_ID))
    train_path = t1.iloc[train_se]["path"].tolist()
    val_path = t1.iloc[val_se]["path"].tolist()
    test_path= t1.iloc[test_se]["path"].tolist()
    np.save('/work/07034/byz/maverick2/GNN/train.npy', np.array(train_ID))
    np.save('/work/07034/byz/maverick2/GNN/val.npy', np.array(val_ID))
    np.save('/work/07034/byz/maverick2/GNN/test.npy', np.array(test_ID))
    np.save('/work/07034/byz/maverick2/GNN/train_path_t.npy', np.array(train_path))
    np.save('/work/07034/byz/maverick2/GNN/val_path_t.npy', np.array(val_path))
    np.save('/work/07034/byz/maverick2/GNN/test_path_t.npy', np.array(test_path))
    return train_path,val_path,test_path

#train,val,test=split_train_val_test(all,[0.5,0.3,0.2],allid)

def read_saved_path(train_path,val_path,test_path):
    train=np.load(train_path)
    val=np.load(val_path)
    test=np.load(test_path)
    train=train.tolist()
    val=val.tolist()
    test=test.tolist()
    return train,val,test

x='/work/07034/byz/maverick2/GNN/train_path_dn_t.npy'
y='/work/07034/byz/maverick2/GNN/val_path_dn_t.npy'
z='/work/07034/byz/maverick2/GNN/test_path_t.npy'
train,val,test=read_saved_path(x,y,z)
train.extend(val)
val=test


def dn_sampling(path,poslist,neglist):
    patient_id = []
    all_path = path
    for i in range(len(all_path)):
        patient_id.append(all_path[i].split("/")[7][0:12])
    t1 = pd.DataFrame({'ID': patient_id, 'path': all_path})
    pos_se = np.where(t1.loc[:, "ID"].isin(poslist))
    neg_se = np.where(t1.loc[:, "ID"].isin(neglist))
    pos_se=list(pos_se)
    neg_se=list(neg_se)
    random.shuffle(pos_se[0])
    random.shuffle(neg_se[0])
    if len(pos_se[0]) > len(neg_se[0]):
        pos_se[0]=pos_se[0][0:len(neg_se[0])]
    else:
        neg_se[0]=neg_se[0][0:len(pos_se[0])]
    left=np.append(pos_se[0],neg_se[0])
    left=[left]
    left=tuple(left)
    pos_se=tuple(pos_se)
    neg_se=tuple(neg_se)
    t1=t1.iloc[left]
    newpath=t1["path"].tolist()
    return newpath

def dn_match(path,number):
    patient_id = []
    all_path = path
    random.shuffle(all_path)
    for i in range(len(all_path)):
        patient_id.append(all_path[i].split("/")[7][0:12])
    t1 = pd.DataFrame({'ID': patient_id, 'path': all_path})
    patient_id=np.unique(patient_id)
    tmp1=pd.DataFrame()
    for j in range(len(patient_id)):
        tmp=t1.loc[t1['ID'] == patient_id[j]]
        if len(tmp) > number:
            tmp=tmp.head(number)
        else:
            tmp=tmp
        tmp1=tmp1.append(tmp)
    out=tmp1["path"].tolist()
    return out

#train=dn_match(train,1000)
#val=dn_match(val,1000)
#test=dn_match(test,100)

#train=dn_sampling(train,POS,NEG)
#val=dn_sampling(val,POS,NEG)
#test=dn_sampling(test,POS,NEG)

match_data(train,POS,NEG)
match_data(val,POS,NEG)
match_data(test,POS,NEG)

#np.save('/work/07034/byz/maverick2/GNN/train_path_dn_t.npy', np.array(train))
#np.save('/work/07034/byz/maverick2/GNN/val_path_dn_t.npy', np.array(val))
#np.save('/work/07034/byz/maverick2/GNN/test_path_dn_m_t_100.npy', np.array(test))

def read_label(path,poslist,neglist):
    label_path=path
    tmplabel=label_path.split("/")[7][0:12]
    if tmplabel in poslist:
        label=1
    elif tmplabel in neglist:
        label=0
    return label

def pat_AUC(model1, data1, NEG1, POS1, type):
    model = copy.deepcopy(model1)
    data = copy.deepcopy(data1)
    NEG = copy.deepcopy(NEG1)
    POS = copy.deepcopy(POS1)
    model.eval()
    model.eval()
    dataset = GraphDataLoader(data, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(2, 2)

    with torch.no_grad():
        predlist = []
        labellist = []
        for inputs, labels in dataset:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            labellist.extend(labels.tolist())
            predlist.extend(predicted.tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(type, 'Accuracy of the network on the images: %d %%' % (
            100 * correct / total))
    print(confusion_matrix)
    print(confusion_matrix.diag() / confusion_matrix.sum(1))
    #################patient level

    patlist = data.graph_list
    pPOS = []
    pNEG = []
    for i in range(len(patlist)):
        if patlist[i][38:50] in NEG:
            pNEG.append(patlist[i][38:50])
        elif patlist[i][38:50] in POS:
            pPOS.append(patlist[i][38:50])
    pNEG = np.unique(pNEG)
    pPOS = np.unique(pPOS)
    pNEG = pNEG.tolist()
    pPOS = pPOS.tolist()
    for i in range(len(patlist)):
        patlist[i] = patlist[i][38:50]
    ID = np.unique(patlist)
    ID = ID.tolist()

    predlabel = []
    truelabel = []
    for i in ID:
        se = []
        # print("processing pat", i)
        for j in range(len(patlist)):
            if patlist[j] == i:
                se.append(j)
        if len(se):
            # print(i, "is in the list")
            pred = [predlist[index] for index in se]
            pro = sum(pred) / len(pred)
            if i in pPOS:
                currtrue = 1
            else:
                currtrue = 0
            truelabel.append(currtrue)
            predlabel.append(pro)


    np.save('/work/07034/byz/maverick2/GNN/truelabel_' + type + '.npy', np.array(truelabel))
    np.save('/work/07034/byz/maverick2/GNN/predlabel_' + type + '.npy', np.array(predlabel))
    np.save('/work/07034/byz/maverick2/GNN/pat_' + type + '.npy', np.array(ID))
    fpr, tpr, thresholds = roc_curve(truelabel, predlabel)
    np.save('/work/07034/byz/maverick2/GNN/tpr_' + type + '.npy', np.array(tpr))
    np.save('/work/07034/byz/maverick2/GNN/fpr_' + type + '.npy', np.array(fpr))
    roc_auc = auc(fpr, tpr)
    print(type, "AUC:", roc_auc)

from dgl.data import DGLDataset
from dgl.data.utils import load_graphs
#sys.path.append(r'/work/07034/byz/maverick2/myapps/site-packages')
#import CGclass
class CGdataset(DGLDataset):

    def __init__(self, graph_path,poslist,neglist):
        self.graph_list=graph_path
        label_list = [read_label(graph_id, poslist, neglist) for graph_id in graph_path]
        self.label_list = label_list

    def __getitem__(self, idx):
        """ get label and graph

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        graph=self.graph_list[idx]
        graph, _ = load_graphs(graph)
        graph=graph[0]
        graph.ndata['feat'] = graph.ndata['feat'][:, 0:512]
        label=self.label_list[idx]
        return graph, label

    def __len__(self):
        """number of graphs"""
        return len(self.graph_list)
### load data-----------------------------
import dgl
from dgl.dataloading import GraphDataLoader
#from torch.utils.data import DataLoader
traindata=CGdataset(train,POS,NEG)
valdata=CGdataset(val,POS,NEG)
testdata=CGdataset(test,POS,NEG)

train_loader = GraphDataLoader(traindata, batch_size=256, shuffle=True,num_workers=8,pin_memory=True)
val_loader = GraphDataLoader(valdata, batch_size=256, shuffle=True,num_workers=8,pin_memory=True)
test_loader = GraphDataLoader(testdata, batch_size=256, shuffle=False,num_workers=8,pin_memory=True)
print("all data loaded")
#print(train_loader.dataloader.dataset.graph_list[2196])
PATH=args.mpath+"1"
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
min_loss=100
non_improve=0
valfre=1600
epochs=args.e
len_train_loss=0
if __name__ == "__main__":
    icounter = 0 #iterations
    for epoch in range(epochs):
        train_loss = 0
        if non_improve == 1000:
            break

        # Training the model
        model.train()
        tcounter = 0
        for inputs, labels in train_loader:
            # Move to device
            inputs=inputs.to(device)
            labels = labels.to(device)
            # Clear optimizers
            optimizer.zero_grad()
            # Forward pass
            output = model(inputs)
            #print('toutput:',output)
            # Loss
            loss = criterion(output, labels)
            # Calculate gradients (backpropogation)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            # Adjust parameters based on gradients
            optimizer.step()
            # Add the loss to the training set's rnning loss
            #print('trainloss:',loss.item())
            train_loss += loss.item() * labels.size(0)
            len_train_loss +=labels.size(0)

            # Print the progress of our training
            tcounter += 1
            icounter += 1
            #print(icounter,"\t",tcounter, "/", len(train_loader))
            if icounter%valfre == 0:
                len_val_loss = 0
                val_loss = 0
                accuracy = 0
                emodel = copy.deepcopy(model)
                emodel.eval()
                counter = 0
                # Tell torch not to calculate gradients
                with torch.no_grad():
                    for einputs, elabels in val_loader:
                        # Move to device
                        einputs, elabels = einputs.to(device), elabels.to(device)
                        # Forward pass
                        output = emodel(einputs)
                        #print('voutput:',output)
                        # Calculate Loss
                        valloss = criterion(output+1e-10, elabels)
                        #print('valloss',valloss.item())
                        # Add loss to the validation set's running loss
                        val_loss += valloss.item() * elabels.size(0)
                        len_val_loss += elabels.size(0)
                        # Since our model outputs a LogSoftmax, find the real
                        # percentages by reversing the log function
                        output = torch.exp(output)
                        # Get the top class of the output
                        top_p, top_class = output.topk(1, dim=1)
                        # See how many of the classes were correct?
                        equals = top_class == elabels.view(*top_class.shape)
                        # Calculate the mean (get the accuracy for this batch)
                        # and add it to the running accuracy for this epoch
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                        # Print the progress of our evaluation
                        counter += 1
                        #print(counter, "/", len(val_loader))

                # Get the average loss for the entire epoch
                train_loss = train_loss/len_train_loss
                print("trainlength: ", len_train_loss,"valength:",len_val_loss)
                len_train_loss=0
                valid_loss = val_loss / len_val_loss
                # Print out the information
                #print("trainlength: ", len_train_loss,"valength:",len_val_loss)
                print('Accuracy: ', accuracy / len(val_loader))
                print(
                    'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tIteration: {}'.format(epoch, train_loss,
                                                                                                     valid_loss,
                                                                                                     icounter))
                if (epoch+1)%20==0:
                    pat_AUC(emodel, traindata, NEG, POS, "train")
                    pat_AUC(emodel, valdata, NEG, POS, "val")
                    torch.save(emodel.state_dict(), args.mpath + "_" + str(epoch + 1))
                #pat_AUC(emodel,traindata,NEG,POS,"train")
                #pat_AUC(emodel, valdata, NEG, POS, "val")
                #pat_AUC(emodel, testdata, NEG, POS, "test")
                if min_loss > valid_loss:
                    min_loss = valid_loss

                if valid_loss > min_loss:
                    non_improve += 1
                    torch.save(emodel.state_dict(), PATH)
                else:
                    non_improve = 0
                torch.save(emodel.state_dict(), args.mpath)
                if non_improve == 1000:
                    print("Early stopping")
                    break


    print('inTest loaded')
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))

    #####accuracy for each class

    classes = ('N', 'A')
    nb_classes = 2
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    print("test acc")
    print(confusion_matrix)
    print(confusion_matrix.diag() / confusion_matrix.sum(1))

    #####accuracy on train set
    train_loader = GraphDataLoader(traindata, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    classes = ('N', 'A')
    nb_classes = 2
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    print("train acc")
    print(confusion_matrix)
    print(confusion_matrix.diag() / confusion_matrix.sum(1))
    #####accuracy on val set
    val_loader = GraphDataLoader(valdata, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    nb_classes = 2
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    print("val acc")
    print(confusion_matrix)
    print(confusion_matrix.diag() / confusion_matrix.sum(1))



    #################pat
    pat_AUC(model,traindata,NEG,POS,"train")
    pat_AUC(model, valdata, NEG, POS, "val")
    pat_AUC(model, testdata, NEG, POS, "test")



