from openai import OpenAI
import json
import numpy as np

client = OpenAI()
# 向chatgpt提問
def ask_gpt3(message):
    response = client.completions.create(
        model = "gpt-3.5-turbo-instruct",
        prompt = message,
        max_tokens=1500
    )
    return response.choices[0].text.strip()

# 計算餘弦相似度的函數
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# 讀取 JSON 檔案
with open('corpus_embeddings.json', 'r') as file:
    data = json.load(file)
print("已讀取corpus_embeddings.json")

while True:
    # 對問題做embedding
    user_question = input("enter your question\n")
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=user_question
    )
    Q_embedding_vector = response.data[0].embedding

    # 取得向量和文字
    vectors = np.array(data['embeddings'])
    labels = data['corpus']
    target_vector = np.array(Q_embedding_vector)
    print("已取得Q_embedding_vector")

    # 計算所有向量與目標向量的相似度
    similarities = [cosine_similarity(vector, target_vector) for vector in vectors]
    print("已計算similarities")

    # 獲取相似度最高的幾組(highest_count)
    highest_count = 2
    top_indices = np.argsort(similarities)[-highest_count:][::-1]
    top_labels = [labels[i] for i in top_indices]
    print("已獲取最高相似度文章")

    # 將最高相似度的文字換行串在一起
    highest_reference = '\n'.join(top_labels)
    print("已串起相似度高文章")
    # print(highest_reference)

    # prompt
    prompt_one = "You are an assistant designed to support medical professionals by providing accurate, concise, and up-to-date information on clinical guidelines and medical protocols. Use technical language appropriately, provide evidence-based recommendations, and cite relevant clinical studies or sources when necessary. Focus on delivering concise yet comprehensive information tailored to medical personnel who require precision and depth in their decision-making process.\n\nFollow these guidelines:\n\n1. **Relevance Check**: Determine if the user's question is related to the given context.\n2. **Responding with References**:\n   - If the question is related, use the context to formulate your answer.\n   - Include references to the specific parts of the context that support your answer.\n3. **Responding without References**:\n   - If the question is not related to the context, answer accurately based on general medical knowledge without mentioning the context.\n4. **Language Matching**: Match the language of your response to the language of the user's question (either English or Traditional Chinese).\n5. **Conciseness and Clarity**: Ensure your responses are clear and concise."
    prompt_two = "Here are the contexts that might be related to the users question."
    prompt_three = "Users question:"
    gpt_input = f"{prompt_one}\n{prompt_two}\n{highest_reference}\n{prompt_three}{user_question}"

    gpt_response = ask_gpt3(gpt_input)
    print("AI: ", gpt_response)
    # print("AI已回答")