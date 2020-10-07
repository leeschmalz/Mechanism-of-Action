import pandas as pd
from pgmpy.models import BayesianModel
from tqdm import tqdm
import numpy as np
from pgmpy.inference import VariableElimination
import random
import networkx as nx
import pylab as plt
from create_network import create_network
from logloss import logloss

df = pd.read_csv('train_features_binary.csv')
df.drop('Unnamed: 0',inplace=True,axis=1)

corr_df = pd.read_csv('feature_correlation.csv')
corr_df.drop('Unnamed: 0',inplace=True,axis=1)

#random train_test_split
train_df = df.sample(int(len(df)*.85))
test_df = df.drop(list(train_df.index),axis=0)
train_df.reset_index(inplace=True,drop=True)
test_df.reset_index(inplace=True,drop=True)

# make model
use_top_corrs = 1000

#get network connections
#network looks like [(node1,node2),...]
network = []
for i in range(len(corr_df[:use_top_corrs])):
    network.append((corr_df['feature1'][i],corr_df['feature2'][i]))

model, nodes_added = create_network(2,2,network)

#model.fit automatically finds cpds for each node
model.fit(train_df)

nx.draw(model, with_labels=True)
plt.show()

evidence = {}
for i in tqdm(range(len(test_df))):
    evidence[i] = {}
    for node in nodes_added:
        evidence[i][node] = test_df[node][i]

#compute variable eliminations for node independencies to speed up model 
infer = VariableElimination(model)

samp_sub = pd.read_csv('sample_submission.csv')

for i in tqdm(range(len(test_df))):
    q = infer.query(variables=['target'],evidence=evidence[i],show_progress=False)
    preds = dict(zip(list(q.state_names['target']),list(q.values)))
    for key in preds.keys():
        samp_sub[key][i] = preds[key]
        samp_sub['sig_id'][i] = test_df['sig_id'][i]


train_targets = pd.read_csv('train_targets_scored.csv')

scores = []
for i in range(len(samp_sub)):
    pred = list(np.array(samp_sub.iloc[i][1:]))
    act = list(np.array(train_targets[train_targets['sig_id'] == samp_sub['sig_id'].iloc[0]].drop('sig_id',axis=1))[0])
    scores.append(logloss(act,pred))
overall_logloss = np.mean(scores)


network = []
for node in nodes_added:
    if len(model.get_children(node)) != 0:
        for child in model.get_children(node):
            network.append((node,child))

#make first log
'''
data = {'Network':  [network],
        'Nodes Count': [len(nodes_added)],
        'Edges Count': [len(network)],
        'Score': [overall_logloss]
        }

logs = pd.DataFrame(data)
logs.to_csv('logs.csv',index=False)
'''

#update logs
logs = pd.read_csv('logs.csv',index_col=False)
logs = logs.append({'Network' : network , 'Nodes Count': len(nodes_added),'Edges Count': len(network),'Score' : overall_logloss} , ignore_index=True)
logs.to_csv('logs.csv',index=False)

logs = pd.read_csv('logs.csv',index_col=False)


