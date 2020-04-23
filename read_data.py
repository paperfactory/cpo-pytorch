import pandas as pd
import numpy as np

def read_file(data_path):
    ''' Load data from train.csv or test.csv. '''

    data = pd.read_csv(data_path, sep=';')
    for col in ['state', 'n_state', 'action_reward']:
        data[col] = [np.array([[np.int(k) for k in ee.split('&')] for ee in e.split('|')]) for e in data[col]]
    for col in ['state', 'n_state']:
        data[col] = [np.array([e[0] for e in l]) for l in data[col]]

    data['action'] = [[e[0] for e in l] for l in data['action_reward']]
    data['reward'] = [tuple(e[1] for e in l) for l in data['action_reward']]
    data.drop(columns=['action_reward'], inplace=True)

    return data

def read_embeddings(embeddings_path):
    ''' Load embeddings (a vector for each item). '''

    embeddings = pd.read_csv(embeddings_path, sep=';')

    return np.array([[np.float64(k) for k in e.split('|')]
                   for e in embeddings['vectors']])

class Embeddings:
    def __init__(self, item_embeddings):
        self.item_embeddings = item_embeddings

    def size(self):
        return self.item_embeddings.shape[1]

    def get_embedding_vector(self):
        return self.item_embeddings

    def get_embedding(self, item_index):
        return self.item_embeddings[item_index]

    def embed(self, item_list):
        return np.array([self.get_embedding(item) for item in item_list])