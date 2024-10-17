from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import json

app = FastAPI()

client = OpenAI()

# Read corpus embeddings once when the server starts
with open('corpus_embeddings.json', 'r') as file:
    data = json.load(file)
print("已讀取corpus_embeddings.json")
vectors = np.array(data['embeddings'])
labels = data['corpus']

# Request body schema
class UserQuestion(BaseModel):
    question: str

# Helper function to call GPT-3 for generating answers
def ask_gpt3(message):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=message,
        max_tokens=1500
    )
    return response.choices[0].text.strip()

# Cosine similarity function
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# API Endpoint for answering questions
@app.post("/ask/")
async def ask_question(user_question: UserQuestion):
    try:
        # Convert question into embedding
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=user_question.question
        )
        Q_embedding_vector = np.array(response.data[0].embedding)

        # Calculate similarities
        similarities = [cosine_similarity(vector, Q_embedding_vector) for vector in vectors]

        # Get top 2 most similar entries
        highest_count = 2
        top_indices = np.argsort(similarities)[-highest_count:][::-1]
        top_labels = [labels[i] for i in top_indices]

        # Prepare the GPT prompt
        prompt_one = """You are an assistant designed to support medical professionals by providing accurate, concise, and up-to-date information on clinical guidelines and medical protocols. Use technical language appropriately, provide evidence-based recommendations, and cite relevant clinical studies or sources when necessary. Focus on delivering concise yet comprehensive information tailored to medical personnel who require precision and depth in their decision-making process.

        Follow these guidelines:

        1. **Relevance Check**: Determine if the user's question is related to the given context.
        2. **Responding with References**:
            - If the question is related, use the context to formulate your answer.
            - Include references to the specific parts of the context that support your answer.
        3. **Responding without References**:
            - If the question is not related to the context, answer accurately based on general medical knowledge without mentioning the context.
        4. **Language Matching**: Match the language of your response to the language of the user's question (either English or Traditional Chinese).
        5. **Conciseness and Clarity**: Ensure your responses are clear and concise."""

        prompt_two = "Here are the contexts that might be related to the users question."
        prompt_three = "Users question:"
        gpt_input = f"{prompt_one}\n{prompt_two}\n{'\n'.join(top_labels)}\n{prompt_three}{user_question.question}"

        # Get the GPT response
        gpt_response = ask_gpt3(gpt_input)

        return {"answer": gpt_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {e}")

# To run the API server, use: uvicorn app:app --reload
