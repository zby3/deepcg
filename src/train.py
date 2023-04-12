import os
import pickle
import yaml
import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader
from histocartography.ml import CellGraphModel
from utils import CGdataset, train, read_label
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float,help='learning rate')
parser.add_argument('--l2', type=float,help='l2 regularization')
parser.add_argument('--mpath', type=str,help='directory to save model')
parser.add_argument('--epoch', type=int,help='epochs')
args = parser.parse_args()

# load train, test path
with open('/byz/E2F4/path.pkl','rb') as f:
    train_path, test_path, POS, NEG = pickle.load(f)

# load data
train_data=CGdataset(train_path,[read_label(tmp, POS, NEG) for tmp in train_path])
test_data=CGdataset(test_path,[read_label(tmp, POS, NEG) for tmp in test_path])
train_loader = GraphDataLoader(train_data, batch_size=256, shuffle=True,num_workers=8,pin_memory=True)
test_loader = GraphDataLoader(test_data, batch_size=256, shuffle=False,num_workers=8,pin_memory=True)

# load model
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

# define hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
epochs = args.epoch

# train
if __name__ == "__main__":
    count = 0  # Epoch counter for non-decreasing val_loss
    for epoch in range(epochs):
        model, _ = train(model, train_loader, optimizer, criterion, device)

        # save model performance per 20 epochs
        if (epoch + 1) % 20 == 0:
            PATH=os.path.join(args.mpath,'_'+str(epoch)+'_'+str(args.lr)+'_'+str(args.l2)+'.pth')
            torch.save(model.state_dict(), PATH)
