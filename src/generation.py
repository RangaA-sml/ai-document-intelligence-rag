import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)


def create_llm():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY was not found.")

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=1024,
        api_key=api_key
    )

    return llm


def check_relevance(llm, query, documents):

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are checking whether the retrieved document content
contains information relevant to the user's question.

Document content:
{context}

User question:
{query}

Answer only with:
YES
or
NO
"""

    response = llm.invoke(prompt)

    result = response.content.strip().upper()

    return result.startswith("YES")


def generate_answer(llm, query, context):

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the provided context.

If the answer is not present in the context, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content