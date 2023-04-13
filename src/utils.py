import os
import pandas as pd
from glob import glob
from PIL import Image
import yaml
import numpy as np
from tqdm import tqdm
import torch
from dgl.data.utils import load_graphs
import argparse
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import copy
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs

class CGdataset(DGLDataset):

    def __init__(self, graph_path,label_list):
        self.graph_list=graph_path
        self.label_list = label_list

    def __getitem__(self, idx):
        graph=self.graph_list[idx]
        graph, _ = load_graphs(graph)
        graph=graph[0]
        graph.ndata['feat'] = graph.ndata['feat'][:, 0:512]
        label=self.label_list[idx]
        return graph, label
    def __len__(self):
        """number of graphs"""
        return len(self.graph_list)

def read_label(path,poslist,neglist,barcode_len):
    label_path=path
    tmplabel=label_path.split("/")[-1][0:barcode_len]
    if tmplabel in poslist:
        label=1
    elif tmplabel in neglist:
        label=0
    return label

def train(model,train_loader,optimizer,criterion,device):
    model.train()
    train_loss = 0
    train_len = 0
    for inputs, labels in train_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)
        # Clear optimizers
        optimizer.zero_grad()
        # Forward pass
        output = model(inputs)
        # Loss
        loss = criterion(output, labels)
        # Calculate gradients (backpropogation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's running loss
        train_loss += loss.item()*labels.size(0)
        train_len += labels.size(0)
    train_loss /= train_len
    return model, train_loss

def pat_auc(model, test_loader, device, barcode_len):
    """
    calculate patient-level AUC on test set
    """
    pred = []  # tile-level prediction
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            pred.extend(predicted.tolist())

    df = pd.DataFrame({'tile': train_loader.dataset.graph_list,
                       'target': train_loader.dataset.label_list})  # summarize the result table
    df['pat'] = df['tile'].apply(lambda x: x.split('/')[-1][0:barcode_len])
    df['pred'] = pred

    # calculate patient-level score (proportion of POS tiles)
    pat=df.pat.unique()
    pat_pred, pat_true=[],[]  #  patient-level scores, true label
    for i in tqdm(len(pat)):
        tmp=df[df['pat']==pat[i]]
        pat_pred.append(tmp['pred'].sum()/len(tmp))
        pat_true.append(tmp['target'].iloc[0])

    # calculate patient-level AUC
    fpr, tpr, thresholds = roc_curve(pat_true, pat_pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc



