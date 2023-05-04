"""
2.1_split-data-hmm
Merge the processed data and generate train and valididation set for GMM-HMM
"""
# %%
data = "/Users/fionshiau/Library/CloudStorage/GoogleDrive-fioncshiau@gmail.com/My Drive/ComputationalGenomics/SignalAproach"
# %%
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('whitegrid')
import numpy as np
import pandas as pd
import os
import glob
# %%
assays = os.listdir(data)
# %%
def merge_norm(idx):
    obs = [pd.read_table(glob.glob(os.path.join(data,i,'*bin*'))[idx],header = None, 
                         names= ['chr','start','end',i]) for i in assays if os.path.isdir(os.path.join(data,i))]
    for i in range(len(obs)):
        obs[i].index = obs[i].loc[:,'chr'].values + '_' + obs[i].loc[:,'start'].values.astype(str) + '-' + obs[i].loc[:,'end'].values.astype(str)
    obs = pd.concat(obs,axis=1)
    obs = obs.loc[:,~obs.columns.duplicated()].copy()
    obs = obs.replace({np.nan:0,'.':0})
    mat = obs.iloc[:,3:].to_numpy()
    from sklearn.preprocessing import normalize
    mat = normalize(X=mat,norm='l2',axis=1)
    obs.iloc[:,3:] = mat
    from sklearn.model_selection import train_test_split
    train_lines, valid_lines = train_test_split(obs.chr.unique(), test_size=0.1, random_state=idx)
    local_path = "/Users/fionshiau/Documents/2023Spring/Computational_Genomics/Final_Project/comp-geno-proj"
    obs.loc[obs.chr.isin(train_lines),:].to_csv(os.path.join(local_path,'data','merged','1.4.{}_train.csv'.format(idx)))
    obs.loc[obs.chr.isin(valid_lines),:].to_csv(os.path.join(local_path,'data','merged','1.4.{}_valid.csv'.format(idx)))
#%%
from tqdm import tqdm
for i in tqdm(range(0,3)):
    merge_norm(i)

