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
from histocartography.ml import CellGraphModel
import argparse
from dgl.data.utils import load_graphs
from tqdm.contrib.concurrent import process_map
from histocartography.interpretability import GraphGradCAMExplainer
from torch.nn import functional as F
from tqdm.contrib.concurrent import process_map
from histocartography.visualization import OverlayGraphVisualization, InstanceImageVisualization


parser = argparse.ArgumentParser(description='E2F4')
parser.add_argument('--outdir', type=str, help='output')
parser.add_argument('--mpath', type=str, help='model path')
parser.add_argument('--dir', type=str, help='input path')
parser.add_argument('--image', type=str, help='image path')
parser.add_argument('--pat', type=str, help='pat')
parser.add_argument('--start', type=int,help='start')
parser.add_argument('--step', type=int,help='step')
parser.add_argument('--index', type=str,help='index')
args = parser.parse_args()

IS_CUDA = torch.cuda.is_available()
# device = 'cuda:0' if IS_CUDA else 'cpu'
device = 'cpu'
NODE_DIM = 512

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
all1="/work/07139/jcm1201/stampede2/baoyi/CPTAC_LUAD_part1_CG/"+args.pat+"*.bin"
all2="/work/07139/jcm1201/stampede2/baoyi/CPTAC_LUAD_part2_CG/"+args.pat+"*.bin"
all3="/work/07034/byz/maverick2/GNN/CPTAC_LUAD_part3_CG/"+args.pat+"*.bin"
all1=glob(all1)
all2=glob(all2)
all3=glob(all3)
myinf=all1+all2+all3

myimage= args.image
#myinf = "/mount/ictr1/chenglab/bzhang/GNN/CPTAC_LUAD_part1_CG/C3N-02155*.bin"
#myinf=myinf[0:100]

visualizer = OverlayGraphVisualization(
        instance_visualizer=InstanceImageVisualization(),
        colormap='jet',
        node_style="fill"
    )

explainer = GraphGradCAMExplainer(model=model)
myout=args.outdir
def get_explain(path):
    global model
    global explainer
    global visualizer
    global myout
    global myimage
    graph, _ = load_graphs(path)
    graph = graph[0]
    graph.ndata['feat'] = graph.ndata['feat'][:, 0:512]
    importance_scores, _ = explainer.process(graph)
    node_attrs = {
        "color": importance_scores
    }
    image_path=os.path.join(myimage,path.split("/")[-1][:-4]+".png")
    try:
        image = np.array(Image.open(image_path))
    except:
        print(path,"no image")
    else:
        canvas = visualizer.process(image, graph, node_attributes=node_attrs)
        path = path.split("/")[-1][:-4]
        canvas.save(os.path.join(myout, path + '_explainer' + ".png"))





if __name__ == '__main__':
    result = process_map(get_explain, myinf, max_workers=16)
    print("succeed")




