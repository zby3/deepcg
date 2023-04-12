import os
import numpy as np
import pandas as pd
from glob import glob
from random import sample
from sklearn.model_selection import train_test_split
import pickle

# load E2F4 scores and patient ID
myinf = "/byz/E2F4/TCGA_LUAD_E2F4_iRAS.txt"
df = pd.read_csv(myinf1,sep="\t")
e2f4_ID = set(df.index)

# load corresponding cell graph paths
cg_path = glob(os.path.join(args.input+'*.bin'))
cg_id = set([tmp.split('/')[-1][:12] for tmp in cg_path])

# find samples with both cell graphs and E2F4 score available
pat_id = e2f4_ID.intersection(cg_id)
df = df.loc[pat_id]
POS = set(df[df["all.intersection.ES"] > 0].index)
NEG = set(df[df["all.intersection.ES"] <= 0].index)

# split train test
train,test=train_test_split(list(pat_id), test_size = 0.3)
train_path=[tmp for tmp in cg_path if tmp.split('/')[-1][:12] in train]
test_path=[tmp for tmp in cg_path if tmp.split('/')[-1][:12] in test]

# downsample in training set for data balance
POS_train_path=[tmp for tmp in train_path if tmp.split('/')[-1][:12] in POS]
NEG_train_path=[tmp for tmp in train_path if tmp.split('/')[-1][:12] in NEG]
if len(POS_train_path) > len(NEG_train_path):
    POS_train_path = sample(POS_train_path,len(NEG_train_path))
else:
    NEG_train_path = sample(NEG_train_path, len(POS_train_path))
train_path = POS_train_path + NEG_train_path

# save train test path
with open('/byz/E2F4/path.pkl', 'wb') as f:
    pickle.dump([train_path, test_path,list(POS),list(NEG)], f)

