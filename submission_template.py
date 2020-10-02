!pip install pgmpy

import pandas as pd
from pgmpy.models import BayesianModel
from tqdm import tqdm
import numpy as np
from pgmpy.inference import VariableElimination
import random
import networkx as nx
import pylab as plt

def create_network(max_parents, max_children, connections):
    
    '''
    Takes in max parents / chilldren per node constraints and network definition (connections). 
    Outputs model and list of nodes included in network.
    '''
    
    model = BayesianModel([connections[0]])
    nodes_added = []
    
    for edge in connections[1:]:
            #whichever has more children 
            #Node A
            try:
                if edge[0] in nodes_added and edge[1] in nodes_added: #both already in network
                    if len(model.get_children(edge[0])) < max_children and len(model.get_children(edge[1])) < max_children:
                        if len(model.get_parents(edge[0])) < max_parents and len(model.get_parents(edge[1])) < max_parents:
                            add_first = random.choice([0,1])
                            model.add_edge(edge[add_first],edge[1-add_first])

                        if len(model.get_parents(edge[0])) < max_parents and len(model.get_parents(edge[1])) >= max_parents:
                            model.add_edge(edge[1],edge[0])

                        if len(model.get_parents(edge[0])) >= max_parents and len(model.get_parents(edge[1])) < max_parents:
                            model.add_edge(edge[0],edge[1])

                    if len(model.get_children(edge[0])) < max_children and len(model.get_children(edge[1])) >= max_children and len(model.get_parents(edge[1])) < max_parents:
                        model.add_edge(edge[0],edge[1])

                    if len(model.get_children(edge[0])) >= max_children and len(model.get_children(edge[1])) < max_children and len(model.get_parents(edge[0])) < max_parents:
                        model.add_edge(edge[1],edge[0])

                elif edge[0] in nodes_added and edge[1] not in nodes_added: #Node A in network Node B not in network
                    if len(model.get_children(edge[0])) < max_children:
                        model.add_edge(edge[0],edge[1])
                        nodes_added.append(edge[1])

                elif edge[0] not in nodes_added and edge[1] in nodes_added: #Node A not in network Node B in network
                    if len(model.get_children(edge[1])) < max_children:
                        model.add_edge(edge[1],edge[0])
                        nodes_added.append(edge[0])

                else:
                    #neither in network, choose randomly
                    add_first = random.choice([0,1])
                    model.add_edge(edge[add_first],edge[1-add_first])
                    nodes_added.append(edge[add_first])
                    nodes_added.append(edge[1-add_first])

            except ValueError: #catch non-DAG error
                pass

    #if node is leaf, connect to target
    for node in nodes_added:
        if node in model.get_leaves():
            model.add_edge(node,'target')
            
    return model, nodes_added

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#--------UNCOMMENT FOR KAGGLE NOTEBOOK----------

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
test_df = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
samp_sub = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

#--------UNCOMMENT FOR KAGGLE NOTEBOOK----------

#--------UNCOMMENT FOR LOCAL RUN----------------
'''
train_features = pd.read_csv('train_features.csv')
test_df = pd.read_csv('test_features.csv')
train_targets_scored = pd.read_csv('train_targets_scored.csv')
samp_sub = pd.read_csv('sample_submission.csv')
'''
#--------UNCOMMENT FOR LOCAL RUN-----------------

#Constants
use_top_corrs = 2000
max_parents = 2
max_children = 2

#make corr_df
feature_1 = []
feature_2 = []
correlation = []
print('making corr_df...')
for col1 in train_features.columns[4:]:
    for col2 in train_features.columns[4:]:
        if col1 != col2:
            if abs(train_features[col1].corr(train_features[col2])) > 0.3:
                feature_1.append(col1)
                feature_2.append(col2)
                correlation.append(abs(train_features[col1].corr(train_features[col2])))
                
corr_df = pd.DataFrame(list(zip(feature_1, feature_2, correlation)), columns =['feature1', 'feature2','correlation']) 
corr_df['abs_correlation'] = abs(corr_df['correlation']) 
corr_df = corr_df.sort_values(by='abs_correlation', ascending=False) #sort by highest abs(correlation)
corr_df.drop('abs_correlation',axis=1,inplace=True)
corr_df = corr_df.iloc[::2, :] #drop every other, duplicates
corr_df = corr_df.reset_index(drop=True)
print('corr_df complete')

#make train_df
cols = list(train_features.columns)
cols.append('target')
train_df = pd.DataFrame(columns=cols)

print('making train_df...')
for i in range(len(train_targets_scored)):
    for col in train_targets_scored.columns:
        if train_targets_scored.iloc[i][col] == 1:
            new_row = dict(train_features.iloc[i])
            new_row['target'] = col
            #add instance to train_df
            train_df = train_df.append(new_row, ignore_index=True)
    
for col in tqdm(list(train_df.columns)[4:-1]):
    for i in range(len(train_df)):
        if train_df[col].iloc[i] > 0:
            train_df[col].iloc[i] = 1
        else: 
            train_df[col].iloc[i] = 0
            
train_df[list(train_df.columns)[4:-1]] = train_df[list(train_df.columns)[4:-1]].astype(int)
print('train_df complete')

#make test_df
print('making test_df...')
for col in list(test_df.columns)[4:]:
    for i in range(len(test_df)):
        if test_df[col].iloc[i] > 0:
            test_df[col].iloc[i] = 1
        else: 
            test_df[col].iloc[i] = 0
            
test_df[list(test_df.columns)[4:]] = test_df[list(test_df.columns)[4:]].astype(int)
print('test_df complete')


#build network
print('building network...')

#get network connections
#network looks like [(node1,node2),...]
connections = []
for i in range(len(corr_df[:use_top_corrs])):
    connections.append((corr_df['feature1'][i],corr_df['feature2'][i]))

model, nodes_added = create_network(max_parents=max_parents, max_children=max_children, connections=connections)

print('network built')

#fit
print('fitting network...')
#model.fit automatically finds cpds for each node
model.fit(train_df)

#compute variable eliminations for node independencies to speed up model 
infer = VariableElimination(model)
print('network fit')

print('making evidence...')
evidence = {}
for i in range(len(test_df)):
    evidence[i] = {}
    for node in nodes_added:
        evidence[i][node] = test_df[node][i]
print('evidence made')

print('predicting probabilities...')
for i in tqdm(range(len(test_df))):
    q = infer.query(variables=['target'],evidence=evidence[i],show_progress=False)
    preds = dict(zip(list(q.state_names['target']),list(q.values)))
    for key in preds.keys():
        samp_sub[key][i] = preds[key]
        samp_sub['sig_id'][i] = test_df['sig_id'][i]

#set max pred to 1 and scale others appropriately
print('scaling...')
for i in range(len(samp_sub)):
    max_pred = max(list(samp_sub.iloc[i])[1:])
    multiplier = 1 / max_pred
    for col in samp_sub.columns[1:]:
        samp_sub[col].iloc[i] = samp_sub[col].iloc[i]*multiplier
print('scaled')

samp_sub.to_csv('submission.csv',index=False)
print('submission csv created')