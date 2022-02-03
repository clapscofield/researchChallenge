#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
CONTENT-BASED RECOMMENDER

Recommendation Systems 2020.2
PPGCC / UFMG
Author: Camila Laranjeira (2020656790)
E-mail: camilalaranjeira@ufmg.br
"""

import pandas as pd
import numpy as np
import argparse, ast

DICT, TEXT = 0, 1

parser = argparse.ArgumentParser(description='Recommendation Systems 2020.2 PPGCC/UFMG by 2020656790.')
parser.add_argument('filepaths', nargs=3, help='Filepaths for ratings.csv and targets.csv')

args = parser.parse_args()
filepaths = args.filepaths

content_filename, train_filename, test_filename = filepaths 


# ### Load train ratings and filter users with few interactions

# In[2]:


def drop( ratings, idx, column, thres):
        
    values = [ui.split(':')[idx] for ui in train_ratings['UserId:ItemId']]
    ratings[column] = values

    values, counts = np.unique(values, return_counts=True)
    drop_values    = [v for k, v in enumerate(values) if counts[k] <= thres]
    ratings        = ratings.drop(ratings[ratings[column].isin(drop_values)].index)

    values = [v for v in values if v not in drop_values] 
    return ratings.drop(column, axis=1),   {v: k  for k, v in enumerate( sorted(values) ) } 


train_ratings = pd.read_csv(train_filename)
train_ratings, users = drop(train_ratings, 0, 'users',20)


# ### Load pre-processed content
# Pre-processing consists in converting terms into lower case and splitting genre categories `(sep = ',')`

# In[3]:


lines = open(content_filename, encoding='utf-8').read().split('\n')

cols = lines[0].split(',')
content = {}

vocabulary = []
vocab_df   = {}
items = {}
for k, line in enumerate(lines[1:-1]):
    idx_comma = line.find(',')

    item = line[:idx_comma]  
    items[item] = k
    attributes  = ast.literal_eval(line[idx_comma+1:])
    
    ### DICTIONARY OF ATTRIBUTES
    content[item] = []
    content[item].append(attributes)
    
    if 'Genre' not in attributes: 
        description = ['n/a']
    else:
        description = [att.lower().strip() for att in attributes['Genre'].split(',')]
    
    content[item].append(description)
        
    for term in description:
        if term not in vocab_df:
            vocab_df[term] = 1
        else:
            vocab_df[term] += 1


# ## Calcutate TF_IDF Matrix `(Items x Vocabulary)`

# In[4]:


keys2idx = {v:k for k,v in enumerate(vocab_df) }

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

for k, i in enumerate(train_ratings.index):
    user_item = train_ratings.loc[i, 'UserId:ItemId'].split(':')
    utility[users.get(user_item[0]), items.get(user_item[1])] = train_ratings.loc[i, 'Prediction'] 


# ## Build user vectors `(Users x Vocabulary)`

# In[6]:


user_vectors = []

for k,user in enumerate(users):
    
    indices = np.nonzero( ~np.isnan(utility[users[user]]) )[0]
    
    user_vector = np.asarray([utility[users[user], idx] * TF_IDF[idx] for idx in indices]) 
    user_vector = (1/len(indices)) * user_vector.sum(axis=0)
    
    user_vectors.append(user_vector)    


# ## Calculate user stats to incorporate into similarity computation

# In[7]:


user_mean = np.nanmean(utility, axis=1)
user_max  = np.nanmax(utility, axis=1)
user_min  = np.nanmin(utility, axis=1)

user_range = [user_min - user_mean, user_max - user_min]


# ## Compute user-item similarity 
# Cosine similarity `cossim` is computed. Then, our estimation is calculates as 
# $\hat{r_{ui}} = \bar{r_u} + r^{min}_{u} + (r^{max}_{u}-r^{min}_{u}) \times cossim $ 
# 
# with $r^{min}_{u}$ and $r^{max}_{u}$ being the deviation from the mean for a user's minimum and maximum ratings respectively. 

# In[8]:


def get_similarity(user, item):
    user_vector = user_vectors[user]
    item_vector = TF_IDF[item]

    sim = np.dot(user_vector, item_vector)/ ( np.linalg.norm(user_vector)*np.linalg.norm(item_vector) )
    
    # umin + (umax-umin)*sim
    delta = user_range[0][user] + (user_range[1][user]-user_range[0][user])*sim 
    return min(10, max(user_mean[user] + delta,0))
    


# ## Compute stats for user cold start

# In[9]:


item_mean  = np.nanmean(utility, axis=0)
item_std   = np.nanstd(utility, axis=0)
item_count = np.sum(np.isnan(utility), axis=0)

global_mean = np.nanmean(utility)


# ## Test predictions
# 
# We consider three scenarios:
# * User cold start: Since we know nothing about the user, the item's average rating is computed. 
# * When both user and item have no associated interaction: A global average of ratings is computed
# * Known user: Similarity between user and item is estimated.

# In[10]:


test = pd.read_csv(test_filename)

rated_items = np.unique([ui.split(':')[1] for ui in train_ratings['UserId:ItemId']] )
rated_items = {v: k  for k, v in enumerate( sorted(rated_items) ) } 


all_pred = []
kind = []
for i in range(len(test)):
    
    user, item = test.iloc[i]['UserId:ItemId'].split(':')
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

test['Prediction'] = all_pred


# In[12]:


with pd.option_context('display.max_rows', None, 
                       'display.max_columns', None): 
    print(test.to_csv(sep=',', index=False))

