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
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs
import dgl
from dgl.dataloading import GraphDataLoader

parser = argparse.ArgumentParser(description='E2F4')
parser.add_argument('--outdir', type=str, help='output')
parser.add_argument('--mpath', type=str, help='model path')
parser.add_argument('--dir', type=str, help='input path')
args = parser.parse_args()

IS_CUDA = torch.cuda.is_available()
device = 'cuda:0' if IS_CUDA else 'cpu'
#device = 'cpu'
NODE_DIM = 512

config_fname = "/work/07034/byz/maverick2/myapps/patho-quant-explainer/core/config/E2F41.yml"

with open(config_fname, 'r') as file:
    config = yaml.safe_load(file)
model = CellGraphModel(
    gnn_params=config['gnn_params'],
    classification_params=config['classification_params'],
    node_dim=NODE_DIM,
    num_classes=2,
    pretrained=False
).to(device)

mymodel=args.mpath+"*"
mymodel=glob(mymodel)


all1="/work/07139/jcm1201/stampede2/baoyi/CPTAC_LUAD_part1_CG/*.bin"
all2="/work/07139/jcm1201/stampede2/baoyi/CPTAC_LUAD_part2_CG/*.bin"
all3="/work/07034/byz/maverick2/GNN/CPTAC_LUAD_part3_CG/*.bin"
all1=glob(all1)
all2=glob(all2)
all3=glob(all3)
all=all1+all2+all3
myinf1="/work/07034/byz/maverick2/GNN/LUAD_E2F4_iRAS_T.txt"
info=pd.read_csv(myinf1,sep="\t")
sam1=np.where(info.loc[:,"all.intersection.ES"]>0)
sam2=np.where(info.loc[:,"all.intersection.ES"]<=0)
info1=info.iloc[sam1]
info2=info.iloc[sam2]
POS=list(info1.index)
NEG=list(info2.index)
allid=POS+NEG

patient_id = []
for i in range(len(all)):
    patient_id.append(all[i].split("/")[7][0:9])
t1 = pd.DataFrame({'ID': patient_id, 'path': all})
se = np.where(t1.loc[:, "ID"].isin(allid))
t1 = t1.iloc[se]
ID = t1["ID"].tolist()
ID = np.unique(ID)
epath=t1["path"].tolist()

def read_label(path,poslist,neglist):
    label_path=path
    tmplabel=label_path.split("/")[7][0:9]
    if tmplabel in poslist:
        label=1
    elif tmplabel in neglist:
        label=0
    return label

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

#######
edata=CGdataset(epath,POS,NEG)
def pat_AUC(model1, data1, NEG1, POS1):
    model = copy.deepcopy(model1)
    data = copy.deepcopy(data1)
    NEG = copy.deepcopy(NEG1)
    POS = copy.deepcopy(POS1)
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
        tmp = patlist[i].split("/")[-1]
        patlist[i] = tmp.split("-")[0] + "-" + tmp.split("-")[1]
    for i in range(len(patlist)):
        if patlist[i] in NEG:
            pNEG.append(patlist[i])
        elif patlist[i] in POS:
            pPOS.append(patlist[i])

    pNEG = np.unique(pNEG)
    pPOS = np.unique(pPOS)
    pNEG = pNEG.tolist()
    pPOS = pPOS.tolist()
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

    fpr, tpr, thresholds = roc_curve(truelabel, predlabel)
    roc_auc = auc(fpr, tpr)
    print("AUC:", roc_auc)

#####
for i in tqdm(range(len(mymodel))):
    model.load_state_dict(torch.load(mymodel[i]))
    print(mymodel[i])
    pat_AUC(model,edata,NEG,POS)
