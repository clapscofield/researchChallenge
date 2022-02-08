#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import argparse, ast

DICT, TEXT = 0, 1

parser = argparse.ArgumentParser(description='Trabalho SR | Clarisse e Bruna')
parser.add_argument('filepaths', nargs=3, help='Inserir entradas na ordem: ratings.jsonl content.jsonl targets.csv')

args = parser.parse_args()
filepaths = args.filepaths

train_filename, content_filename, test_filename = filepaths 


# ### Load train ratings and filter users with few interactions

# In[2]:


def drop(ratings, idx, column, thres):
        
    values = train_ratings.iloc[:, idx]
    ratings[column] = values

    values, counts = np.unique(values, return_counts=True)
    drop_values    = [v for k, v in enumerate(values) if counts[k] <= thres]
    ratings        = ratings.drop(ratings[ratings[column].isin(drop_values)].index)

    values = [v for v in values if v not in drop_values] 
    return ratings.drop(column, axis=1),   {v: k  for k, v in enumerate( sorted(values) ) } 


train_ratings = pd.read_json(train_filename, lines=True) 
train_ratings, users = drop(train_ratings, 0, 'users',20) 

conteudo = pd.read_json(content_filename, lines=True) 

content = {}
vocabulary = []
vocab_df   = {}
items = {}
k = 0

for index, linha in conteudo.iterrows():
    item = linha["ItemId"]
    items[item] = k
    k = k + 1

    atributos  = linha[conteudo.columns.difference(['ItemId'])]
    #dicionario de atributos
    content[item] = []
    content[item].append(atributos)

    if 'Genre' not in atributos: 
        description = ['n/a']
    else:
        description = [att.lower().strip() for att in atributos['Genre'].split(',')]
    
    content[item].append(description)
    
    #contagem de repeticao de termos
    for term in description:
        if term not in vocab_df:
            vocab_df[term] = 1
        else:
            vocab_df[term] += 1
    
## ATEEEE AQUIIII

# ## Calcutate TF_IDF Matrix `(Items x Vocabulary)`

# In[4]:


keys2idx = {v:k for k,v in enumerate(vocab_df)}

TF_IDF = []
for k, item in enumerate(items.keys()):
    
    vector = np.zeros(len(vocab_df)) 
    for term in content[item][TEXT]:
        
        tf  = content[item][TEXT].count(term)
        idf = np.log( len(content)/ vocab_df[term]) 

        vector[keys2idx[term]] = tf * idf

    TF_IDF.append(vector)

TF_IDF = np.asarray(TF_IDF)


# ## Build utility matrix to facilitate further computations

# In[5]:

m, n = len(users), len(items)
utility = np.full((m, n), np.nan)

#k eh contagem
#i eh o valor do index
#user item[0] eh o usuario e user item[1] eh o item
for index, linha in train_ratings.iterrows():
    usuario_id = linha['UserId']
    item_id = linha['ItemId']
    utility[users.get(usuario_id), items.get(item_id)] = linha['Rating']

# # ## Build user vectors `(Users x Vocabulary)`

# # In[6]:


user_vectors = []

for k, user in enumerate(users):
    
    indices = np.nonzero( ~np.isnan(utility[users[user]]) )[0]
    
    user_vector = np.asarray([utility[users[user], idx] * TF_IDF[idx] for idx in indices]) 
    user_vector = (1/len(indices)) * user_vector.sum(axis=0)
    
    user_vectors.append(user_vector)    



user_mean = np.nanmean(utility, axis=1)
user_max  = np.nanmax(utility, axis=1)
user_min  = np.nanmin(utility, axis=1)

user_range = [user_min - user_mean, user_max - user_min]



def get_similarity(user, item):
    user_vector = user_vectors[user]
    item_vector = TF_IDF[item]

    sim = np.dot(user_vector, item_vector)/ ( np.linalg.norm(user_vector)*np.linalg.norm(item_vector) )
    
    # umin + (umax-umin)*sim
    delta = user_range[0][user] + (user_range[1][user]-user_range[0][user])*sim 
    return min(10, max(user_mean[user] + delta,0))


item_mean  = np.nanmean(utility, axis=0)
item_std   = np.nanstd(utility, axis=0)
item_count = np.sum(np.isnan(utility), axis=0)

global_mean = np.nanmean(utility)

## AAAAAAAAAAAAAAAAH

teste = pd.read_csv(test_filename) 

rated_items = np.unique([ui for ui in train_ratings['ItemId']]) 
rated_items = {v: k  for k, v in enumerate(sorted(rated_items))} 


all_pred = []
kind = []
for i in range(len(teste)):
    user = teste.iloc[i]['UserId']
    item = teste.iloc[i]['ItemId']
    user = users.get(user)  
    rated_item = rated_items.get(item) 
    item = items.get(item)
    
    if user is None and rated_item is None :
        kind.append('cold')
        pred = global_mean
        
    elif user is None:
        kind.append('cold')
        pred = item_mean[item] - 1.65 * (item_std[item]/item_count[item])
                    
    else:
        kind.append('sim')
        pred = get_similarity(user, item)
    
    all_pred.append(global_mean if np.isnan(pred) else pred)

teste['Rating'] = all_pred
teste = teste.groupby('UserId', as_index=False).apply(lambda x: x.sort_values('Rating', ascending=False))
teste = teste.drop('Rating', axis=1)

# In[12]:


with pd.option_context('display.max_rows', None, 
                       'display.max_columns', None): 
    print(teste.to_csv(sep=',', index=False))

