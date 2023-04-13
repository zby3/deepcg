import pandas as pd
from glob import glob
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_ind,ranksums

mydir="/byz/E2F4/interpretation/*.csv"
myinf=glob(mydir)

def cal_stats(dat1,dat2,num):
    wp = ranksums(dat1.iloc[:, (num)], dat2.iloc[:, (num)]).pvalue
    tp = ttest_ind(dat1.iloc[:, (num)], dat2.iloc[:, (num)]).pvalue
    t = ttest_ind(dat1.iloc[:, (num)], dat2.iloc[:, (num)]).statistic
    ratio=np.mean(dat1.iloc[:, (num)]) / np.mean(dat2.iloc[:, (num)])
    return wp,tp,t,ratio

df = []
for i in tqdm(range(len(myinf))):
    info1 = pd.read_csv(myinf[i], index_col=0)
    info1 = info1[~info1.isin([np.nan, np.inf, -np.inf]).any(1)]

    info1t = info1[info1['importance'] > 0.7]
    info1f = info1[info1['importance'] < 0.1]

    tmp=[]
    for j in range(24):
        l=j+2
        r1, r2, r3, r4 = cal_stats(info1t, info1f, l)
        tmp.extend([r1, r2, r3, r4])
    df.append(tmp)

df = np.array(df)
names=['wp','tp','t','ratio']
feat=["area","convex_area","eccentricity","equivalent_diameter","euler_number",
                "extent","filled_area","major_axis_length","minor_axis_length","orientation",
                "perimeter","solidity","roughness","shape_factor","ellipticity","roundness",
                "glcm_contrast","glcm_dissimilarity","glcm_homogeneity","glcm_energy",
                "glcm_ASM","glcm_dispersion","mean_crowdedness","std_crowdedness"]
names_col=[]
for i in range(len(feat)):
    cur=[tmp+"_"+feat[i] for tmp in names]
    names_col.extend(cur)

df = pd.DataFrame(df, columns=names_col,index=[tmp.split('/')[-1][:-4] for tmp in myinf])
df.to_csv('/byz/E2F4/interpretation.csv')