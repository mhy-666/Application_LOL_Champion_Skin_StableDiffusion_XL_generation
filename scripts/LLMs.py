import os
from openai import AzureOpenAI
import streamlit as st

def get_chat_answer(prompt, context, chat_model = "RAG-gpt-35"):
    '''
    This function returns the response from the RAG model and the GPT-3.5 model
    
    Args:
    prompt: str
    context: list
    chat_model: str

    Returns:
    rag_answer: str
    gpt_answer: str

    '''
    
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    RAG_context = "Here are chunks similar to the prompt from extra backstory of League of Legends. You must use them to guide your response as long as they align with the prompt in some degree: \n In the game League of Legends, "
    for i in range(len(context)):
        RAG_context += f" {context[i]}"
    # print(RAG_context)

    messages = [
    {"role": "system", "content": "You are a person who are very familar League of Legends, and you are good at summry from the chunks."},
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": RAG_context}
  ]
    rag_answer =  client.chat.completions.create(
        model = chat_model,
        messages = messages,
        stream = True
    )

    messages = [
    {"role": "system", "content": "You are a gaming encyclopedia."},
    {"role": "user", "content": prompt}
    ]

    gpt_answer =  client.chat.completions.create(
        model = chat_model,
        messages = messages,
        stream = True
    )

    return rag_answer, gpt_answer