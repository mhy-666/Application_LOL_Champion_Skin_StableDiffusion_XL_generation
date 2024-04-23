import streamlit as st
from PIL import Image
import io
import requests
from openai import AzureOpenAI
import os
import re


# OpenAI API配置
client = AzureOpenAI(
        api_key=st.secrets["AZURE_OPENAI_KEY"],  
        api_version=st.secrets["OPENAI_API_VERSION"],
        azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"]
    )

# Hugging Face API配置
import requests

API_URL = "https://api-inference.huggingface.co/models/wintercoming6/lol-champion-skin-sdxl-lora3"
headers = {"Authorization": f"Bearer {st.secrets['HF_API_TOKEN']}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content
image_bytes = query({
	"inputs": "Astronaut riding a horse",
})

# 定义函数,根据GPT-3.5的响应生成4张图片
def generate_images(response_chunks):
    images = []
    for chunk in response_chunks:
        image_bytes = query({"inputs": chunk})
        image = Image.open(io.BytesIO(image_bytes))
        images.append((image, chunk))
    return images

def split_into_sentences(text):
    sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', text)
    return sentences

def split_into_paragraphs(text):
    paragraphs = text.split("\n\n")
    return paragraphs

def main():
    # Streamlit应用
    st.set_page_config(page_title="League of Legend Comic Generator", page_icon="📚")
    st.title("League of Legend Comic Generator")
    st.image("./data/league-of-legends-pc-game-cover.jpg")
    
    # 创建文本输入框和提交按钮
    prompt = st.text_input("enter your prompt here:")
    submit_button = st.button("Submit")
    messages = [
    {"role": "system", "content": "You are a writer assistant who helps people write comic plot, you need to write plot in 4 paragraphs every time."},
    {"role": "user", "content": prompt}
    ]
    # 如果用户点击了提交按钮,则发送请求到GPT-3.5
    if submit_button and prompt:
        gpt_answer =  client.chat.completions.create(
        model = 'RAG-gpt-35',
        messages = messages
        )
        response_text = gpt_answer.choices[0].message.content
        


        paragraphs = split_into_paragraphs(response_text)
        response_chunks = paragraphs  # 只取前4个段落


        # 使用 GPT-3.5 的响应生成 4 张图片
        images = generate_images(response_chunks)

        st.session_state['images'] = images

    # 添加翻页按钮
    if 'images' in st.session_state:
        images = st.session_state['images']
        prev_page, next_page = st.columns(2)
        current_page = st.session_state.get("current_page", 0)
        if prev_page.button("Previous Page"):
            current_page = max(0, current_page - 1)
        if next_page.button("Next Page"):
            current_page = min(len(images) - 1, current_page + 1)
        st.session_state["current_page"] = current_page
        st.image(images[current_page][0], caption=f"Page {current_page+1}", use_column_width=True)
        st.write(images[current_page][1])


if __name__ == "__main__":
    main()
