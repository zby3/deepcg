import sys

sys.path.remove('/opt/apps/intel18/impi18_0/python2/2.7.15/lib/python2.7/site-packages')
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
from tqdm.contrib.concurrent import process_map

parser = argparse.ArgumentParser(description='E2F4')
parser.add_argument('--outdir', type=str, help='output')
parser.add_argument('--mpath', type=str, help='model path')
parser.add_argument('--dir', type=str, help='input path')
args = parser.parse_args()

IS_CUDA = torch.cuda.is_available()
# device = 'cuda:0' if IS_CUDA else 'cpu'
device = 'cpu'
NODE_DIM = 512

config_fname = "/work/07034/byz/maverick2/myapps/patho-quant-explainer/core/config/E2F41.yml"

with open(config_fname, 'r') as file:
    config = yaml.safe_load(file)
model = CellGraphModel(
    gnn_params=config['gnn_params'],
    classification_params=config['classification_params'],
    node_dim=512,
    num_classes=2,
    pretrained=False
).to(device)

model.load_state_dict(torch.load(args.mpath, map_location=torch.device('cpu')))
# model=model.to('cpu')
all1 = "/work/07139/jcm1201/stampede2/baoyi/CPTAC_LUAD_part1_CG/*.bin"
all2 = "/work/07139/jcm1201/stampede2/baoyi/CPTAC_LUAD_part2_CG/*.bin"
all3 = "/work/07034/byz/maverick2/GNN/CPTAC_LUAD_part3_CG/*.bin"
all1 = glob(all1)
all2 = glob(all2)
all3 = glob(all3)
myinf= all1 + all2 + all3

#myinf = args.dir + "*.bin"
#myinf = glob(myinf)
#myinf=myinf[0:100]
model.eval()

def predict(path):
    global model
    graph, _ = load_graphs(path)
    graph = graph[0]
    graph.ndata['feat'] = graph.ndata['feat'][:, 0:512]
    outputs = model(graph)
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.numpy()
    predicted = predicted[0]
    concepts = np.array([path,predicted])
    return concepts

out = args.outdir


if __name__ == '__main__':
    result = process_map(predict, myinf, max_workers=16)
    result = np.vstack(result)
    print(result.shape)
    df = pd.DataFrame(result)
    df.columns=["file","prediction"]
    df.to_csv(path_or_buf=out, sep=',')
    print("succeed")
