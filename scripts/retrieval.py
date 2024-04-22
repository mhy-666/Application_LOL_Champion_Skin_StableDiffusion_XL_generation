import numpy as np
import pandas as pd
import os

from src.embedding import get_embedding

from pinecone import Pinecone

def cosine_similarity(a, b):
    return np.dot(a, b)

def topk_retrieval(ids:list, query:str, index, k:int=5):
    '''
    This function returns the top k similar chunks to the query
    
    Args:
    ids: list
    query: str
    index: Pinecone.index
    k: int
    
    Returns:
    topk_similarity: list
    topk_metadata: list
    '''
    topk_similarity = []
    topk_metadata = []
    query_vector = get_embedding(query)
    for name in ids:
        records = index.fetch([name])
        similarity = cosine_similarity(query_vector, records.vectors[name]['values'])
        topk_similarity.append(similarity)
        topk_metadata.append(records.vectors[name]['metadata']['text'])
    
    topk_similarity, topk_metadata = zip(*sorted(zip(topk_similarity, topk_metadata), reverse=True))
    topk_similarity = list(topk_similarity[:k])
    topk_metadata = list(topk_metadata[:k])
    
    return topk_similarity, topk_metadata

def get_chunk_ids(csv_path:str='../data/champion_names.csv'):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_name_path = os.path.join(parent_dir, 'data/champion_names.csv')
    
    ids = pd.read_csv(csv_name_path)
    ids = ids['champion_with_number'].values.tolist()
    return ids

if __name__ == '__main__':
    ids = get_chunk_ids(csv_path = '../data/champion_names.csv')
    query = ids[0]
    res_topk_similarity, res_topk_metadata = topk_retrieval(query, k=5, ids=ids)