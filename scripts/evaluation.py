import os
from openai import AzureOpenAI
from scripts.retrieval import topk_retrieval, get_chunk_ids
from scripts.data_storage import get_index

class Evaluation:
    def __init__(self):
        self.database = get_index()
        self.champion_list = ['Ahri', 'Akali', 'Wukong', 'Teemo', 'Volibear', 'Yasuo', 'Lux', 'Zoe', 'Jhin', 'Leesin', 'Leblanc', 'Caitlyn']
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

    def get_rag_and_no_rag_answers(self):
        RAG_answer = []
        No_RAG_answer = []
        for champion in self.champion_list:
            prompt = f"Write a comic plot about {champion} in league of legend."
            _, context = topk_retrieval(
                ids=get_chunk_ids(), query=prompt, index=self.database, k=5
            )
            rag, gpt = self.get_chat_answer(prompt, context, chat_model='RAG-gpt-35')
            RAG_answer.append(rag)
            No_RAG_answer.append(gpt)
        return RAG_answer, No_RAG_answer

    def get_chat_answer(self, prompt, context, chat_model="RAG-gpt-35"):
        client = AzureOpenAI(
            api_key=os.getenv("64e7c187bf824fb6a1ed25d889592335"),
            api_version=os.getenv("2023-12-01-preview"),
            azure_endpoint=os.getenv("https://ragembedding.openai.azure.com/")
        )
        RAG_context = "Here are chunks similar to the prompt from extra backstory of League of Legends. You must use them to guide your response as long as they align with the prompt in some degree: \n In the game League of Legends, "
        for i in range(len(context)):
            RAG_context += f" {context[i]}"

        messages = [
            {"role": "system", "content": "You are a person who are very familar League of Legends, and you are good at summry from the chunks."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": RAG_context}
        ]
        rag_answer = client.chat.completions.create(
            model=chat_model,
            messages=messages,
            stream=True
        )

        messages = [
            {"role": "system", "content": "You are a gaming encyclopedia."},
            {"role": "user", "content": prompt}
        ]

        gpt_answer = client.chat.completions.create(
            model=chat_model,
            messages=messages,
            stream=True
        )

        return rag_answer, gpt_answer

    def evaluate_comic_plots(self, RAG_answer, No_RAG_answer):
        results = []
        for rag, gpt in zip(RAG_answer, No_RAG_answer):
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful League of Legends comic plots critic."},
                    {"role": "user", "content": "Which of the following 2 comic plot is most close to League of Legend's style? Give me the index, like the first/second."
                     + "1. " + rag + '\n'
                     + "2. " + gpt
                    }
                ]
            )
            print(response.choices[0].message.content)
            results.append(response.choices[0].message.content)
        return results
