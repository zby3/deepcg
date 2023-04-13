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
from histocartography.interpretability import GraphGradCAMExplainer
from torch.nn import functional as F
from tqdm.contrib.concurrent import process_map


parser = argparse.ArgumentParser(description='E2F4')
parser.add_argument('--outdir', type=str, help='output')
parser.add_argument('--mpath', type=str, help='model path')
parser.add_argument('--dir', type=str, help='input path')
parser.add_argument('--start', type=int,help='start')
parser.add_argument('--step', type=int,help='step')
parser.add_argument('--index', type=str,help='index')
parser.add_argument('--e2f4', type=str,help='file of e2f4 and pred scores')
parser.add_argument('--percent', type=float,help='top/bottom e2f4 scores')
parser.add_argument('--threshold', type=float,help='threshold for prediction')
parser.add_argument('--DIM', type=int,help='feature dimension')
parser.add_argument('--POS', type=int,help='1 for POS, 0 for NEG')

args = parser.parse_args()

percent=args.percent
threshold=args.threshold

info=pd.read_csv(args.e2f4,sep="\t")
pat=info.index.tolist()

IS_CUDA = torch.cuda.is_available()
# device = 'cuda:0' if IS_CUDA else 'cpu'
device = 'cpu'
NODE_DIM = args.DIM

#config_fname = "/mount/ictr1/chenglab/bzhang/GNN/config/E2F41.yml"
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
mymodel=args.mpath
#mymodel="/mount/ictr1/chenglab/bzhang/GNN/models/4_16/E2F4_lr_0.0002_l2_0.pt"
model.load_state_dict(torch.load(mymodel, map_location=torch.device('cpu')))
# model=model.to('cpu')
myinf=[]
for i in range(len(pat)):
    myin = args.dir + pat[i] + "*.bin"
    myin = glob(myin)
    myinf.extend(myin)

print("length:",len(myinf))

#myinf = "/mount/ictr1/chenglab/bzhang/GNN/CPTAC_LUAD_part1_CG/C3N-02155*.bin"
#myinf=myinf[0:100]
#myinf=myinf[args.start:(args.start+args.step)]
#print("current length",len(myinf))
explainer = GraphGradCAMExplainer(model=model)
def get_explain(path):
    global model
    global explainer
    graph, _ = load_graphs(path)
    graph = graph[0]
    graph.ndata['feat'] = graph.ndata['feat'][:, 0:512]
    concepts = graph.ndata["concepts"]
    g = copy.deepcopy(graph)
    out = model(g)
    out = F.softmax(out, dim=1).data.squeeze()
    out = out.cpu().tolist()[1]
    out = np.repeat(out, concepts.shape[0])
    out = np.reshape(out, (out.size, 1))

    tmp=path.split("/")

    img = np.repeat(tmp[-1], concepts.shape[0])
    img = np.reshape(img, (img.size, 1))
    importance_scores, _ = explainer.process(graph)
    importance_scores = np.reshape(importance_scores, (importance_scores.size, 1))

    res = np.append(importance_scores, out, axis=1)
    res = np.append(res, concepts, axis=1)
    res = np.append(res, img, axis=1)
    return res

myout=args.outdir


if __name__ == '__main__':
    myinf=myinf[args.start:(args.start+args.step)]
    result = process_map(get_explain, myinf, max_workers=16)
    result = np.vstack(result)
    print(result.shape)
    df = pd.DataFrame(result)
    # df.columns=["file","prediction"]
    cuout = myout+"_"+ args.index + ".csv"
    df.to_csv(path_or_buf=cuout, sep=',')
    print("succeed")




