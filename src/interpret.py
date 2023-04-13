import os
import pandas as pd
from glob import glob
import yaml
import numpy as np
import torch
from histocartography.ml import CellGraphModel
import copy
from dgl.data.utils import load_graphs
from histocartography.interpretability import GraphGradCAMExplainer
from torch.nn import functional as F
from tqdm.contrib.concurrent import process_map
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str,help='directory of CPTAC cell graphs')
parser.add_argument('--mpath', type=str,help='path of the saved model')
parser.add_argument('--output', type=str,help='directory to save importance scores and cellular features')
args = parser.parse_args()

# load cell graph paths
cg_path = glob(os.path.join(args.input+'*.bin'))
cg_id = list(set([tmp.split('/')[-1][:9] for tmp in cg_path]))

# load model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
config_fname="./config/E2F4.yml"
with open(config_fname, 'r') as file:
    config = yaml.safe_load(file)
model = CellGraphModel(
    gnn_params=config['gnn_params'],
    classification_params=config['classification_params'],
    node_dim=NODE_DIM,
    num_classes=2,
    pretrained=False
).to(device)
model.load_state_dict(torch.load(args.mpath))

# define explainer
explainer = GraphGradCAMExplainer(model=model)

def get_explain(path):
    global model
    global explainer
    graph, _ = load_graphs(path)
    graph = graph[0]
    graph.ndata['feat'] = graph.ndata['feat'][:, 0:512]
    concepts = graph.ndata["concepts"]

    # get tile id
    tile = np.repeat(path.split('/')[-1], concepts.shape[0])
    tile = np.reshape(tile, (tile.size, 1))

    # get probability
    g = copy.deepcopy(graph)
    out = model(g)
    out = F.softmax(out, dim=1).data.squeeze()
    out = out.cpu().tolist()[1]
    out = np.repeat(out, concepts.shape[0])
    out = np.reshape(out, (out.size, 1))

    # get importance score for each cell
    importance_scores, _ = explainer.process(graph)
    importance_scores = np.reshape(importance_scores, (importance_scores.size, 1))

    # summarize results
    res = np.append(importance_scores, out, axis=1)
    res = np.append(res, concepts, axis=1)
    res = np.append(res, tile, axis=1)
    return res

if __name__ == '__main__':
    for i in range(len(cg_id)):
        myinf = [tmp for tmp in cg_path if tmp.split[-1][:9] == cg_id[i]]
        result = process_map(get_explain, myinf, max_workers=16)
        result = np.vstack(result)
        df = pd.DataFrame(result)
        df.columns = ["importance", "prediction", "area", "convex_area", "eccentricity", "equivalent_diameter",
                      "euler_number",
                      "extent", "filled_area", "major_axis_length", "minor_axis_length", "orientation",
                      "perimeter", "solidity", "roughness", "shape_factor", "ellipticity", "roundness",
                      "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy",
                      "glcm_ASM", "glcm_dispersion", "mean_crowdedness", "std_crowdedness", "tile"]
        df.to_csv(path_or_buf=os.path.join(args.output,cg_id[i]+'.csv'), sep=',')

