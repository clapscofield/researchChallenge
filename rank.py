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

teste = pd.read_csv('teste2.csv') 
tentativa = teste.groupby('UserId', as_index=False).apply(lambda x: x.sort_values('Rating', ascending=False))
print(tentativa.to_csv(sep=',', index=False))