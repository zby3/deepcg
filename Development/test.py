import os
import pickle
import torch
import yaml
from dgl.dataloading import GraphDataLoader
from histocartography.ml import CellGraphModel
from glob import glob
from tqdm import tqdm
from utils import CGdatset, read_label, pat_AUC
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mpath', type=str,help='directory of saved models')
args = parser.parse_args()

# load train, test path
with open('/byz/E2F4/path.pkl','rb') as f:
    train_path, test_path, POS, NEG = pickle.load(f)

# load data
train_data=CGdataset(train_path,[read_label(tmp, POS, NEG) for tmp in train_path])
test_data=CGdataset(test_path,[read_label(tmp, POS, NEG) for tmp in test_path])
train_loader = GraphDataLoader(train_data, batch_size=256, shuffle=True,num_workers=8,pin_memory=True)
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

# load model path
mpath=os.path.join(args.mpath,"*")
mpath=glob(mpath)

# evaluate models
barcode_len = 12 # set 12 for TCGA cohort and 9 for CPTAC cohort
best_auc = 0.5

for i in tqdm(range(len(mpath))):
    model.load_state_dict(torch.load(mpath[i]))
    auc=pat_AUC(model, test_loader, device, barcode_len)
    if auc > best_auc:
        best_auc = auc
        best_model = mpath[i].split('/')[-1]

f"{best_model}: {best_auc}"
