#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import argparse, ast

DICT, TEXT = 0, 1

parser = argparse.ArgumentParser(description='Trabalho SR | Clarisse e Bruna')
parser.add_argument('arquivos', nargs=3, help='Inserir entradas na ordem: ratings.jsonl content.jsonl targets.csv')

args = parser.parse_args()
arquivos = args.arquivos

arquivo_treino, arquivo_conteudo, arquivo_teste = arquivos 


# Carrega os ratings de treino e filtra os usuários
def drop(ratings, idx, coluna, arvores):
        
    valores = ratings_treino.iloc[:, idx]
    ratings[coluna] = valores

    valores, counts = np.unique(valores, return_counts=True)
    drop_values = [v for k, v in enumerate(valores) if counts[k] <= arvores]
    ratings = ratings.drop(ratings[ratings[coluna].isin(drop_values)].index)

    valores = [v for v in valores if v not in drop_values] 
    return ratings.drop(coluna, axis=1),   {v: k  for k, v in enumerate( sorted(valores) ) } 


ratings_treino = pd.read_json(arquivo_treino, lines=True) 
ratings_treino, usuarios = drop(ratings_treino, 0, 'users',20) 

conteudo_arquivo = pd.read_json(arquivo_conteudo, lines=True) 

conteudo = {}
vocabulario = []
vocab_df   = {}
itens = {}
k = 0

for index, linha in conteudo_arquivo.iterrows():
    item = linha["ItemId"]
    itens[item] = k
    k = k + 1

    atributos  = linha[conteudo_arquivo.columns.difference(['ItemId'])]
    #dicionario de atributos
    conteudo[item] = []
    conteudo[item].append(atributos)

    if 'Genre' not in atributos: 
        descricao = ['n/a']
    else:
        descricao = [att.lower().strip() for att in atributos['Genre'].split(',')]
    
    conteudo[item].append(descricao)
    
    #contagem de repeticao de termos
    for term in descricao:
        if term not in vocab_df:
            vocab_df[term] = 1
        else:
            vocab_df[term] += 1
    
## ATEEEE AQUIIII

# ## Calcutate TF_IDF Matrix `(Items x Vocabulary)`


chaves_idx = {v:k for k,v in enumerate(vocab_df)}

TF_IDF = []
for k, item in enumerate(itens.keys()):
    
    vetor = np.zeros(len(vocab_df)) 
    for term in conteudo[item][TEXT]:
        
        tf  = conteudo[item][TEXT].count(term)
        idf = np.log( len(conteudo)/ vocab_df[term]) 

        vetor[chaves_idx[term]] = tf * idf

    TF_IDF.append(vetor)

TF_IDF = np.asarray(TF_IDF)


# Constroi matriz para facilitar próximas computações
m, n = len(usuarios), len(itens)
matriz = np.full((m, n), np.nan)

#k eh contagem
#i eh o valor do index
#user item[0] eh o usuario e user item[1] eh o item
for index, linha in ratings_treino.iterrows():
    usuario_id = linha['UserId']
    item_id = linha['ItemId']
    matriz[usuarios.get(usuario_id), itens.get(item_id)] = linha['Rating']


# Constroi vetores de usuários (Usuarios x Vocabularios)
vetores_usuarios = []

for k, usuario in enumerate(usuarios):
    
    indices = np.nonzero( ~np.isnan(matriz[usuarios[usuario]]) )[0]
    
    user_vector = np.asarray([matriz[usuarios[usuario], idx] * TF_IDF[idx] for idx in indices]) 
    user_vector = (1/len(indices)) * user_vector.sum(axis=0)
    
    vetores_usuarios.append(user_vector)    

media_usuarios = np.nanmean(matriz, axis=1)
max_usuario  = np.nanmax(matriz, axis=1)
min_usuario  = np.nanmin(matriz, axis=1)

range_usuario = [min_usuario - media_usuarios, max_usuario - min_usuario]


def similaridade(user, item):
    vetor_usuario = vetores_usuarios[user]
    vetor_item = TF_IDF[item]

    sim = np.dot(vetor_usuario, vetor_item)/ ( np.linalg.norm(vetor_usuario)*np.linalg.norm(vetor_item) )

    # umin + (umax-umin)*sim
    delta = range_usuario[0][user] + (range_usuario[1][user]-range_usuario[0][user])*sim 
    return min(10, max(media_usuarios[user] + delta,0))


media_item  = np.nanmean(matriz, axis=0)
std_item   = np.nanstd(matriz, axis=0)
soma_item = np.sum(np.isnan(matriz), axis=0)

media_global = np.nanmean(matriz)

## AAAAAAAAAAAAAAAAH

teste = pd.read_csv(arquivo_teste) 

items_rated = np.unique([ui for ui in ratings_treino['ItemId']]) 
items_rated = {v: k  for k, v in enumerate(sorted(items_rated))} 


predicoes = []
tipo = []
for i in range(len(teste)):
    usuario = teste.iloc[i]['UserId']
    item = teste.iloc[i]['ItemId']
    usuario = usuarios.get(usuario)  
    item_rated = items_rated.get(item) 
    item = itens.get(item)
    
    if usuario is None and item_rated is None :
        tipo.append('cold')
        pred = media_global
        
    elif usuario is None:
        tipo.append('cold')
        pred = media_item[item] - 1.65 * (std_item[item]/soma_item[item])
                    
    else:
        tipo.append('sim')
        pred = similaridade(usuario, item)
    
    predicoes.append(media_global if np.isnan(pred) else pred)

teste['Rating'] = predicoes
teste = teste.groupby('UserId', as_index=False).apply(lambda x: x.sort_values('Rating', ascending=False))
teste = teste.drop('Rating', axis=1)

with pd.option_context('display.max_rows', None, 
                       'display.max_columns', None): 
    print(teste.to_csv(sep=',', index=False))

