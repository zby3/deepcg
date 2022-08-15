import sys
sys.path.remove('/opt/apps/intel18/impi18_0/python2/2.7.15/lib/python2.7/site-packages')
import os
import pandas as pd
from glob import glob
import yaml
import numpy as np
import torch
from histocartography.ml import CellGraphModel
import argparse
import copy
from dgl.data.utils import load_graphs
from tqdm.contrib.concurrent import process_map
import random
from torch.nn import functional as F
parser = argparse.ArgumentParser(description='E2F4')
parser.add_argument('--outdir', type=str, help='output')
parser.add_argument('--mpath', type=str, help='model path')
parser.add_argument('--start', type=int,help='start')
parser.add_argument('--step', type=int,help='step')
parser.add_argument('--index', type=str,help='index')
parser.add_argument('--trained', type=int,help='whether to use trained weights or random weights, > 0 for trained')
parser.add_argument('--DIM', type=int,help='feature dimension')


args = parser.parse_args()

###load CG
all1 = "/work/07139/jcm1201/stampede2/baoyi/CPTAC_LUAD_part1_CG/*.bin"
all2 = "/work/07139/jcm1201/stampede2/baoyi/CPTAC_LUAD_part2_CG/*.bin"
all3 = "/work/07034/byz/maverick2/GNN/CPTAC_LUAD_part3_CG/*.bin"
all1 = glob(all1)
all2 = glob(all2)
all3 = glob(all3)
myinf= all1 + all2 + all3

#random.seed(30)
#myinf = random.sample(myinf,100000)

###model
IS_CUDA = torch.cuda.is_available()
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
if args.trained > 0:
    model.load_state_dict(torch.load(mymodel, map_location=torch.device('cpu')))


def get_representation(path):
    global model
    global explainer
    graph, _ = load_graphs(path)
    graph = graph[0]
    graph.ndata['feat'] = graph.ndata['feat'][:, 0:512]
    g = copy.deepcopy(graph)
    embeddings = model.cell_graph_gnn(g, g.ndata['feat'])
    embeddings=embeddings.cpu().detach().numpy()
    
    out = model(graph)
    out = F.softmax(out, dim=1).data.squeeze()
    out = out.cpu().tolist()[1]
    out = np.repeat(out, embeddings.shape[0])
    out = np.reshape(out, (out.size, 1))
    
    tmp=path.split("/")

    img = np.repeat(tmp[-1], embeddings.shape[0])
    img = np.reshape(img, (img.size, 1))
    res = np.append(embeddings, img, axis=1)
    res = np.append(res, out, axis=1)
    return res

myout=args.outdir


if __name__ == '__main__':
    myinf=myinf[args.start:(args.start+args.step)]
    result = process_map(get_representation, myinf, max_workers=16)
    result = np.vstack(result)
    print(result.shape)
    df = pd.DataFrame(result)
    # df.columns=["file","prediction"]
    cuout = myout+"_" + args.index + ".csv"
    df.to_csv(path_or_buf=cuout, sep=',')
    print("succeed")
