import os
import numpy as np
import pandas as pd
from glob import glob
import torch
import yaml
from dgl.dataloading import GraphDataLoader
from histocartography.ml import CellGraphModel
from glob import glob
from utils import CGdatset, read_label, pat_AUC
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str,help='directory of CPTAC cell graphs')
parser.add_argument('--mpath', type=str,help='path of the saved model')
args = parser.parse_args()

# load E2F4 scores and patient ID
myinf = "/byz/E2F4/CPTAC_LUAD_E2F4_iRAS.txt"
df = pd.read_csv(myinf1,sep="\t")
e2f4_ID = set(df.index)

# load corresponding cell graph paths
cg_path = glob(os.path.join(args.input+'*.bin'))
cg_id = set([tmp.split('/')[-1][:9] for tmp in cg_path])

# find samples with both cell graphs and E2F4 score available
pat_id = e2f4_ID.intersection(cg_id)
df = df.loc[pat_id]
POS = set(df[df["all.intersection.ES"] > 0].index)
NEG = set(df[df["all.intersection.ES"] <= 0].index)

# load data
test_path=[tmp for tmp in cg_path if tmp.split('/')[-1][:9] in pat_id]
test_data=CGdataset(test_path,[read_label(tmp, POS, NEG, 9) for tmp in test_path])
test_loader = GraphDataLoader(test_data, batch_size=256, shuffle=False,num_workers=8,pin_memory=True)

# load model architecture
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
config_fname="./config/E2F4.yml"
with open(config_fname, 'r') as file:
    config = yaml.safe_load(file)
model = CellGraphModel(
        gnn_params=config['gnn_params'],
        classification_params=config['classification_params'],
        node_dim=512,
        num_classes=2,
        pretrained=False
    ).to(device)

# evaluate the model on independent data
model.load_state_dict(torch.load(args.mpath))
auc=pat_AUC(model, test_loader, device, 9)
f"AUC: {auc}"

