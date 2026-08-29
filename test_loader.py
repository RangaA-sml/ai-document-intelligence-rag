from src.document_loader import load_pdf
from src.chunking import split_documents
from src.embeddings import create_embedding_model
from src.vector_store import create_vector_store
from src.retriever import retrieve_documents
from src.generation import create_llm, generate_answer


pdf_path = "data/documents/attention_is_all_you_need.pdf"


# 1. Load PDF
documents = load_pdf(pdf_path)

# 2. Split into chunks
chunks = split_documents(documents)

# 3. Create embedding model
embedding_model = create_embedding_model()

# 4. Create vector store
vector_store = create_vector_store(
    chunks,
    embedding_model
)

# 5. Create LLM
llm = create_llm()


# 6. User question
query = "What is self-attention?"


# 7. Retrieve relevant chunks
results = retrieve_documents(
    vector_store,
    query,
    k=3
)


# 8. Combine retrieved chunks
context = "\n\n".join(
    result.page_content
    for result in results
)


# Generate answer
answer = generate_answer(
    llm,
    query,
    context
)

print("\n==============================")
print("QUESTION")
print("==============================")
print(query)

print("\n==============================")
print("ANSWER")
print("==============================")
print(answer)

print("\n==============================")
print("SOURCES")
print("==============================")

for i, result in enumerate(results, start=1):

    print(f"\nSource {i}")
    print("File:", result.metadata.get("source"))
    print("Page:", result.metadata.get("page_label"))