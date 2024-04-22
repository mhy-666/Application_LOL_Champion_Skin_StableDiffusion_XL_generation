from src.data_extraction import load_data
from src.data_chunk import chunk_story
from src.embedding import get_df_embeddings

import pandas as pd
from pinecone import Pinecone
import os


    
def data_processing(csv_path:str='../data/champions_lore.csv'):
    '''
    This function processes the data and returns a DataFrame with the processed data

    '''
    df = load_data(csv_path = csv_path)
    new_df = chunk_story(df, chunk_size=1000, overlap_size=200)
    new_df['champion'] = new_df['champion'].str.replace("[’\s]", "", regex=True)
    new_df['champion_with_number'] = new_df['champion'] + (new_df.groupby('champion').cumcount() + 1).astype(str)
    
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_name_path = os.path.join(parent_dir, 'data/champion_names.csv')
    
    new_df['champion_with_number'].to_csv(csv_name_path, index=False)

    final_df = get_df_embeddings(new_df, model='RAG-Embedding')

    # Create a new DataFrame with the processed data
    processed_df = pd.DataFrame({'id': new_df['champion_with_number'], 'values': final_df, 'metadata': new_df['story_chunk']})

    
    # Add metadata to the processed_df
    for _, row in processed_df.iterrows():
        index = row.name
        story_chunk = row['metadata']

        metadata_dict = {}

        metadata_dict['chunk'] = index
        metadata_dict['text'] = story_chunk
        row['metadata'] = metadata_dict

    return processed_df

def store_data():
    '''
    This function stores the processed data into pinecone vector database and returns the index
    '''
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(parent_dir, 'data/champions_lore.csv')
    processed_df = data_processing(csv_path=csv_path)

    # initialize connection to pinecone (get API key at app.pc.io)
    api_key = os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY'
    environment = os.environ.get('PINECONE_ENVIRONMENT') or 'PINECONE_ENVIRONMENT'

    # configure client
    pc = Pinecone(api_key=api_key)

    # create a new index
    index_name = 'rag'
    index = pc.Index(index_name)

    index.upsert_from_dataframe(processed_df, batch_size=100)
    print(f"Data stored in index {index_name}")

    return index

def get_index():
    '''
    This function returns the index
    '''
    index_name = 'rag'
    api_key = os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY'
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    return index

    
    