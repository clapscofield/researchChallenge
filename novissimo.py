import pandas as pd
import numpy as np
import argparse

DICT, TEXT = 0, 1

class Recomendador():
    def __init__(self, treino, conteudo, teste):

        self.conteudo_arquivo = conteudo
        self.teste = teste
        self.treino = treino
        self.treino, self.usuarios = self.drop(treino, 0, 'users',20) 
        self.conteudo = {}
        self.itens = {}

    def drop(self, ratings, idx, coluna, arvores):
        valores = ratings.iloc[:, idx]
        ratings[coluna] = valores

        valores, counts = np.unique(valores, return_counts=True)
        drop_values = [v for k, v in enumerate(valores) if counts[k] <= arvores]
        ratings = ratings.drop(ratings[ratings[coluna].isin(drop_values)].index)

        valores = [v for v in valores if v not in drop_values] 
        return ratings.drop(coluna, axis=1),   {v: k  for k, v in enumerate( sorted(valores) ) }

    def combina_features(self, row):
        row['Genre_str'] = ' '.join([str(elem) for elem in row['Genre']])
        row['Actors_str'] = ' '.join([str(elem) for elem in row['Actors']])

        return row['Genre_str'] +' '+row['Title']+' '+row['Director']+' '+ row['Actors_str']

    def inicia(self):
        vocab_df = {}
        k = 0 
        features = ['Genre', 'Title', 'Actors', 'Director']

        #preprocess
        for feature in features:
            conteudo_arquivo[feature] = conteudo_arquivo[feature].fillna('')

        conteudo_arquivo['combina_features'] = conteudo_arquivo.apply(self.combina_features, axis = 1)

        for index, linha in conteudo_arquivo.iterrows():
            item = linha["ItemId"]
            self.itens[item] = k
            k = k + 1

            atributos  = linha[conteudo_arquivo.columns.difference(['ItemId'])]

            #dicionario de atributos
            self.conteudo[item] = []
            self.conteudo[item].append(atributos)

            if 'combina_features' not in atributos: 
                descricao = ['n/a']
            else:
                descricao = [att.lower().strip() for att in atributos['combina_features'].split(',')]
            
            self.conteudo[item].append(descricao)
            
            #contagem de repeticao de termos
            for term in descricao:
                if term not in vocab_df:
                    vocab_df[term] = 1
                else:
                    vocab_df[term] += 1
        return vocab_df

    def calcula_TF_IDF(self, vocab_df):
        chaves_idx = {v:k for k,v in enumerate(vocab_df)}

        TF_IDF = []
        for k, item in enumerate(self.itens.keys()):
            
            vetor = np.zeros(len(vocab_df)) 
            for term in self.conteudo[item][TEXT]:
                
                tf  = self.conteudo[item][TEXT].count(term)
                idf = np.log( len(self.conteudo)/ vocab_df[term]) 

                vetor[chaves_idx[term]] = tf * idf

            TF_IDF.append(vetor)

        TF_IDF = np.asarray(TF_IDF)
        return TF_IDF

    def constroi_matriz_utilidade(self):
        # Constroi matriz para facilitar próximas computações
        m, n = len(self.usuarios), len(self.itens)
        matriz = np.full((m, n), np.nan)

        for index, linha in self.treino.iterrows():
            usuario_id = linha['UserId']
            item_id = linha['ItemId']
            matriz[self.usuarios.get(usuario_id), self.itens.get(item_id)] = linha['Rating']
        return matriz

    def constroi_vetor_usuarios(self, matriz, TF_IDF):
        vetor_usuarios = []

        for k, usuario in enumerate(self.usuarios):
            
            indices = np.nonzero( ~np.isnan(matriz[self.usuarios[usuario]]) )[0]
            
            user_vector = np.asarray([matriz[self.usuarios[usuario], idx] * TF_IDF[idx] for idx in indices]) 
            user_vector = (1/len(indices)) * user_vector.sum(axis=0)
            
            vetor_usuarios.append(user_vector)
        return vetor_usuarios

    def similaridade(self, TF_IDF, user, item, vetor_usuarios, range_usuario, media_usuarios):
        vetor_usuario = vetor_usuarios[user]
        vetor_item = TF_IDF[item]

        sim = np.dot(vetor_usuario, vetor_item)/ ( np.linalg.norm(vetor_usuario)*np.linalg.norm(vetor_item) )

        # umin + (umax-umin)*sim
        delta = range_usuario[0][user] + (range_usuario[1][user]-range_usuario[0][user])*sim 
        return min(10, max(media_usuarios[user] + delta,0))

    def predicao(self):
        vocab_df = self.inicia()
        TF_IDF = self.calcula_TF_IDF(vocab_df)
        matriz = self.constroi_matriz_utilidade()
        vetor_usuarios = self.constroi_vetor_usuarios(matriz, TF_IDF)

        media_usuarios = np.nanmean(matriz, axis=1)
        max_usuario  = np.nanmax(matriz, axis=1)
        min_usuario  = np.nanmin(matriz, axis=1)
        range_usuario = [min_usuario - media_usuarios, max_usuario - min_usuario]

        media_item  = np.nanmean(matriz, axis=0)
        std_item   = np.nanstd(matriz, axis=0)
        soma_item = np.sum(np.isnan(matriz), axis=0)

        media_global = np.nanmean(matriz)

        teste = pd.read_csv(arquivo_teste)
        items_rated = np.unique([ui for ui in self.treino['ItemId']]) 
        items_rated = {v: k  for k, v in enumerate(sorted(items_rated))} 
        predicoes = []
        tipo = []
        for i in range(len(teste)):
            usuario = teste.iloc[i]['UserId']
            item = teste.iloc[i]['ItemId']
            usuario = self.usuarios.get(usuario)  
            item_rated = items_rated.get(item) 
            item = self.itens.get(item)
            
            if usuario is None and item_rated is None :
                tipo.append('cold')
                pred = media_global
                
            elif usuario is None:
                tipo.append('cold')
                pred = media_item[item] - 10 * (std_item[item]/soma_item[item])
                            
            else:
                tipo.append('sim')
                pred = self.similaridade(TF_IDF, usuario, item, vetor_usuarios, range_usuario, media_usuarios)
            
            predicoes.append(media_global if np.isnan(pred) else pred)

        teste['Rating'] = predicoes
        teste = teste.groupby('UserId', as_index=False).apply(lambda x: x.sort_values('Rating', ascending=False))
        teste = teste.drop('Rating', axis=1)
        with pd.option_context('display.max_rows', None, 
                       'display.max_columns', None): 
            print(teste.to_csv(sep=',', index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trabalho SR | Clarisse e Bruna')
    parser.add_argument('arquivos', nargs=3, help='Inserir entradas na ordem: ratings.jsonl content.jsonl targets.csv')
    args = parser.parse_args()

    arquivos = args.arquivos
    arquivo_treino, arquivo_conteudo, arquivo_teste = arquivos 

    treino = pd.read_json(arquivo_treino, lines=True) 
    conteudo_arquivo = pd.read_json(arquivo_conteudo, lines=True) 

    recomendador = Recomendador(treino, conteudo_arquivo, arquivo_teste)
    recomendador.predicao()


