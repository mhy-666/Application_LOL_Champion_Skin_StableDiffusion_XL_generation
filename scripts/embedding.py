from openai import AzureOpenAI   
import os
import streamlit as st

def get_embedding(text, model="RAG-Embedding"):
   '''
   This function returns the embeddings of the text using the RAG-Embedding model
   
   Args:
   text: str
   model: str
   
   Returns:
   embedding: list
   '''

   client = AzureOpenAI(
        api_key=st.secrets["AZURE_OPENAI_KEY"],  
        api_version=st.secrets["OPENAI_API_VERSION"],
        azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"]
    )
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_df_embeddings(df, model="RAG-Embedding"):
    '''
    This function returns the embeddings of the DataFrame using the RAG-Embedding model
    
    Args:
    df: DataFrame
    model: str
    
    Returns:
    final_df: DataFrame

    '''
    final_df = df['story_chunk'].apply(lambda x: get_embedding(x, model=model))
    return final_df