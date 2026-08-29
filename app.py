import streamlit as st

from src.document_loader import load_pdf
from src.chunking import split_documents
from src.embeddings import create_embedding_model
from src.vector_store import create_vector_store
from src.retriever import retrieve_documents
from src.generation import (
    create_llm,
    generate_answer,
    check_relevance
)


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Document Intelligence RAG",
    page_icon="📄",
    layout="wide"
)


# -----------------------------------
# Sidebar - Project Information
# -----------------------------------

with st.sidebar:

    st.title("📄 About")

    st.write(
        "This AI-powered application allows users "
        "to interact with PDF documents using "
        "Retrieval-Augmented Generation (RAG)."
    )

    st.markdown("### 🔹 This application can:")

    st.markdown(
        """
        - 📄 Process PDF documents
        - 🧩 Split documents into meaningful chunks
        - 🧠 Generate text embeddings
        - 🔎 Perform semantic similarity search
        - 🤖 Generate answers using an LLM
        - 📚 Show relevant source pages
        - 🛡️ Reduce answers unrelated to the document
        """
    )

    st.markdown("---")

    st.markdown("### ⚙️ Technologies")

    st.markdown(
        """
        **Python**

        **Streamlit**

        **LangChain**

        **FAISS**

        **Groq LLM**

        **HuggingFace Embeddings**
        """
    )

    st.markdown("---")

    st.markdown("### 🔄 RAG Pipeline")

    st.markdown(
        """
        PDF  
        ↓  
        Text Extraction  
        ↓  
        Chunking  
        ↓  
        Embeddings  
        ↓  
        FAISS Vector Store  
        ↓  
        Semantic Retrieval  
        ↓  
        Relevance Check  
        ↓  
        Groq LLM  
        ↓  
        Answer + Sources
        """
    )

    st.markdown("---")

    st.caption(
        "Built with Python, LangChain, FAISS, "
        "Streamlit and Groq."
    )


# -----------------------------------
# Session State
# -----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------------
# Main Application
# -----------------------------------

st.title("📄 Document Intelligence RAG")

st.write(
    "Upload a PDF and ask questions about its content."
)


# -----------------------------------
# PDF Upload
# -----------------------------------

uploaded_file = st.file_uploader(
    "📤 Choose a PDF file",
    type=["pdf"]
)


if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    if st.button(
        "🚀 Process PDF",
        type="primary"
    ):

        with st.spinner(
            "Processing document..."
        ):

            # Load PDF
            documents = load_pdf(
                uploaded_file
            )

            # Split into chunks
            chunks = split_documents(
                documents
            )

            # Create embedding model
            embedding_model = create_embedding_model()

            # Create vector store
            vector_store = create_vector_store(
                chunks,
                embedding_model
            )

            # Create LLM
            llm = create_llm()

            # Store objects in session
            st.session_state.vector_store = vector_store
            st.session_state.llm = llm
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.num_pages = len(documents)
            st.session_state.num_chunks = len(chunks)

            # Clear previous conversation
            st.session_state.messages = []

        st.success(
            "✅ PDF processed successfully!"
        )


# -----------------------------------
# Document Information + Chat
# -----------------------------------

if "vector_store" in st.session_state:

    st.divider()

    st.subheader("📑 Document Information")

    st.write(
        f"**File:** {st.session_state.pdf_name}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Pages",
            st.session_state.num_pages
        )

    with col2:

        st.metric(
            "Text Chunks",
            st.session_state.num_chunks
        )

    st.divider()

    # -----------------------------------
    # Chat History
    # -----------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

            if message.get("sources"):

                st.caption("📚 Sources")

                for page in message["sources"]:

                    st.write(
                        f"Page {page}"
                    )


    # -----------------------------------
    # Ask Question
    # -----------------------------------

    query = st.chat_input(
        "Ask something about the document..."
    )


    if query:

        # Display user question
        with st.chat_message("user"):

            st.write(query)


        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching document..."
            ):

                # Retrieve relevant chunks
                results = retrieve_documents(
                    st.session_state.vector_store,
                    query,
                    k=3
                )


                # -----------------------------------
                # Retrieval Check
                # -----------------------------------

                if not results:

                    answer = (
                        "I could not find the answer "
                        "in the provided document."
                    )

                    relevant = False
                    pages = []


                else:

                    # Check whether retrieved chunks
                    # are relevant to the question
                    relevant = check_relevance(
                        st.session_state.llm,
                        query,
                        results
                    )


                    if not relevant:

                        answer = (
                            "I could not find the answer "
                            "in the provided document."
                        )

                        pages = []


                    else:

                        # Combine retrieved chunks
                        context = "\n\n".join(
                            result.page_content
                            for result in results
                        )


                        # Generate answer
                        answer = generate_answer(
                            st.session_state.llm,
                            query,
                            context
                        )


                        # -----------------------------------
                        # Collect Source Pages
                        # -----------------------------------

                        pages = []

                        for result in results:

                            page = result.metadata.get(
                                "page_label"
                            )

                            if page not in pages:

                                pages.append(page)


            # -----------------------------------
            # Display Answer
            # -----------------------------------

            st.write(answer)


            # -----------------------------------
            # Display Sources
            # -----------------------------------

            if relevant and pages:

                st.caption("📚 Sources")

                for i, page in enumerate(
                    pages,
                    start=1
                ):

                    st.write(
                        f"Source {i} — Page {page}"
                    )


        # -----------------------------------
        # Save Conversation
        # -----------------------------------

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })


        assistant_message = {
            "role": "assistant",
            "content": answer
        }


        if relevant and pages:

            assistant_message["sources"] = pages


        st.session_state.messages.append(
            assistant_message
        )


        st.rerun()


    # -----------------------------------
    # Clear Document
    # -----------------------------------

    st.divider()

    if st.button("🗑️ Clear Document"):

        st.session_state.clear()

        st.rerun()