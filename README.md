# ai-document-intelligence-rag
📚 DocuMind --- Document Intelligence RAG
An AI-powered PDF question-answering application using
Retrieval-Augmented Generation (RAG).

DocuMind lets users upload PDF documents and ask questions in natural
language. It retrieves relevant passages from the uploaded document and
uses a language model to generate context-based answers with source-page
references when available.
Features
- Upload PDF documents through a Streamlit interface.
- Extract text and page information from PDF files.
- Split extracted text into smaller chunks for retrieval.
- Generate semantic embeddings using Hugging Face Sentence
  Transformers.
- Store and search vectors with FAISS.
- Retrieve passages relevant to a user's question.
- Generate context-based answers using a Groq-hosted language model.
- Display source-page references when available.
- Ask questions through an interactive chat interface.
Technology Stack
  Technology                           Purpose
  Python                               Core programming language
  Streamlit                            Web interface
  LangChain                            Document processing and LLM integration
  PyPDF / PyPDFLoader                  PDF text extraction
  Hugging Face Sentence Transformers   Text embeddings
  FAISS                                Vector similarity search
  Groq                                 Hosted language model inference
  python-dotenv                        Environment variable management
How It Works
1. Upload PDF --- The user selects a PDF document.
2. Extract text --- Text and page metadata are extracted.
3. Chunk the document --- Text is split into smaller chunks.
4. Generate embeddings --- Each chunk is converted into a vector
   representation.
5. Index vectors --- FAISS stores the vectors for similarity
   search.
6. Retrieve context --- Relevant chunks are retrieved for the
   user's question.
7. Check relevance --- Retrieved content is checked for relevance.
8. Generate an answer --- The language model uses the question and
   retrieved context to compose a response.
9. Show sources --- Available source-page references are displayed
   with the answer.
Project Structure
document-intelligence-rag/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
└── src/
    ├── __init__.py
    ├── document_loader.py
    ├── chunking.py
    ├── embeddings.py
    ├── vector_store.py
    ├── retriever.py
    └── generation.py
Installation and Setup
1. Clone your repository
git clone <your-repository-url>
cd document-intelligence-rag
Replace <your-repository-url> with your GitHub repository URL.
2. Create a virtual environment
python -m venv venv
Activate it on Windows:
venv\Scripts\activate
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
4. Configure your Groq API key
Create a .env file in the project root and add:
GROQ_API_KEY=your_groq_api_key_here
Get an API key from the Groq Console.

Security: Do not commit your real .env file or API key to GitHub.
Keep .env in .gitignore.
5. Run the application
streamlit run app.py
Open the local URL printed in the terminal if the app does not open
automatically.
How to Use
1. Open the application.
2. Navigate to Ask your PDF.
3. Upload a PDF document.
4. Select Process PDF and wait for processing to finish.
5. Enter a question in the chat input.
6. Read the generated answer and review any source-page references.
Embedding Model and Language Model
Embedding model
The project uses sentence-transformers/all-MiniLM-L6-v2 to convert
text chunks into vectors for semantic similarity search.
Language model
A Groq-hosted language model is configured in the generation module. The
configured model name and API settings are defined in
src/generation.py.


Limitations and Security
- Generated answers can be inaccurate, even when based on retrieved
  passages.
- Check source pages against the original PDF for important decisions.
- Scanned PDFs may require OCR if their text cannot be extracted
  directly.
- Large documents can require additional memory and processing time.
- API access and model availability depend on the provider's limits.
- Do not upload confidential documents unless your environment is
  approved to process them.
Future Improvements
- Support additional document formats.
- Add OCR for scanned PDFs.
- Improve source citations with page-level excerpts.
- Support multiple documents in one session.
- Add downloadable answers and conversation history.
- Evaluate retrieval quality and answer accuracy.
- Add persistent vector storage for previously processed documents. 


Project Objective
DocuMind aims to make document analysis easier by combining semantic
search with generative AI. Instead of manually searching lengthy PDFs,
users can ask questions and receive answers based on relevant document
content.