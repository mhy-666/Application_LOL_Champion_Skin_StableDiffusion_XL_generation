import streamlit as st
from PIL import Image
import io
import requests
from openai import AzureOpenAI
import os

print(os.getenv("AZURE_OPENAI_KEY"))
# OpenAI API配置
client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

# Hugging Face API配置
import requests

API_URL = "https://api-inference.huggingface.co/models/wintercoming6/lol-champion-skin-sdxl-lora3"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content
image_bytes = query({
	"inputs": "Astronaut riding a horse",
})

# 定义函数,根据GPT-3.5的响应生成4张图片
def generate_images(response):
    images = []
    for i in range(4):
        image_bytes = query({"inputs": response})
        image = Image.open(io.BytesIO(image_bytes))
        images.append(image)
    return images

st.title("League of Legends Comic Book")
# Streamlit应用
st.set_page_config(page_title="漫画书")

# 创建文本输入框
prompt = st.text_input("输入prompt")

# 如果用户输入了prompt,则发送请求到GPT-3.5
if prompt:
    messages = [
    {"role": "system", "content": "You are a gaming encyclopedia."},
    {"role": "user", "content": prompt}
    ]

    gpt_answer =  client.chat.completions.create(
        model = 'RAG-gpt-35',
        messages = messages,
        stream = True
    )
    response_text = gpt_answer.choices[0].text.strip()

    # 使用GPT-3.5的响应生成4张图片
    images = generate_images(response_text)

    # 显示图片
    for i, image in enumerate(images):
        st.image(image, caption=f"第 {i+1} 页", use_column_width=True)

# 添加翻页按钮
if images:
    prev_page, next_page = st.columns(2)
    current_page = st.session_state.get("current_page", 0)
    if prev_page.button("上一页"):
        current_page = max(0, current_page - 1)
    if next_page.button("下一页"):
        current_page = min(len(images) - 1, current_page + 1)
    st.session_state["current_page"] = current_page
    st.image(images[current_page], caption=f"第 {current_page+1} 页", use_column_width=True)