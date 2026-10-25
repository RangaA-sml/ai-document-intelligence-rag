import streamlit as st

from src.document_loader import load_pdf
from src.chunking import split_documents
from src.embeddings import create_embedding_model
from src.vector_store import create_vector_store
from src.retriever import retrieve_documents
from src.generation import create_llm, generate_answer, check_relevance


st.set_page_config(
    page_title="DocuMind | Document Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#17213b; --muted:#65718a; --purple:#7657e8; --teal:#18a999; --line:#e7e9f4; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:linear-gradient(180deg,#f7f7ff 0%,#f9fbff 40%,#ffffff 100%); color:var(--ink); }
    [data-testid="stHeader"] { background:rgba(247,247,255,.88); }
    .block-container { max-width:1180px; padding-top:1.2rem; padding-bottom:3rem; }
    .brandbar { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 18px; background:#fff; border:1px solid var(--line); border-radius:18px; box-shadow:0 8px 28px rgba(43,38,96,.05); margin-bottom:20px; }
    .brandleft { display:flex; align-items:center; gap:12px; }
    .brandicon { width:44px; height:44px; display:grid; place-items:center; border-radius:14px; color:white; font-size:23px; background:linear-gradient(135deg,#7657e8,#4c8ee8); box-shadow:0 7px 16px #7657e833; }
    .brandname { font:800 22px/1.2 'Manrope',sans-serif; letter-spacing:-.6px; color:#17213b !important; text-shadow:none !important; opacity:1 !important; }
    .brandtag { color:#7b849b; font-size:10px; font-weight:700; letter-spacing:1.7px; margin-top:2px; }
    .topbadge { padding:8px 12px; border-radius:999px; color:#168a7d; background:#e8faf6; font-size:11px; font-weight:800; white-space:nowrap; }
    .hero { padding:42px 38px; border-radius:24px; background:radial-gradient(circle at 88% 15%,#b8fff0 0,transparent 25%),linear-gradient(125deg,#201c4b,#3c327a 56%,#4c72bd); color:white; position:relative; overflow:hidden; margin-bottom:26px; }
    .eyebrow { color:#c9fff4; font-size:11px; letter-spacing:2px; font-weight:800; text-transform:uppercase; }
    .hero h1 { font:800 clamp(32px,4.5vw,54px)/1.08 'Manrope',sans-serif; letter-spacing:-1.8px; margin:14px 0; color:#fff; }
    .hero p { color:#e0e5ff; font-size:16px; line-height:1.75; max-width:700px; margin:0; }
    .hero-note { display:inline-flex; margin-top:24px; border:1px solid #ffffff3d; background:#ffffff12; border-radius:999px; padding:9px 13px; color:#f2f4ff; font-size:12px; font-weight:700; }
    .section-kicker { color:#7657e8; text-transform:uppercase; letter-spacing:1.8px; font-size:10px; font-weight:800; margin-bottom:6px; }
    .section-title { font:800 27px 'Manrope',sans-serif; color:#17213b; letter-spacing:-.8px; margin:0 0 6px; }
    .section-copy { color:#65718a; line-height:1.7; font-size:14px; margin-bottom:20px; }
    .feature-card { height:100%; padding:22px; background:#fff; border:1px solid var(--line); border-radius:18px; box-shadow:0 8px 28px #28245308; }
    .feature-icon { width:43px; height:43px; display:grid; place-items:center; border-radius:13px; background:#f0edff; color:#7657e8; font-size:21px; margin-bottom:16px; }
    .feature-card h3 { color:#202849; font:700 16px 'Manrope',sans-serif; margin:0 0 8px; }
    .feature-card p { color:#68748c; font-size:13px; line-height:1.7; margin:0; }
    .process-step { padding:18px; border-radius:16px; background:#fff; border:1px solid var(--line); min-height:120px; }
    .step-number { color:#7657e8; font:800 12px 'Manrope',sans-serif; background:#f0edff; display:inline-block; border-radius:8px; padding:7px 9px; margin-bottom:11px; }
    .process-step h4 { margin:0 0 6px; color:#202849; font:700 14px 'Manrope',sans-serif; }
    .process-step p { margin:0; color:#68748c; font-size:12px; line-height:1.65; }
    div[data-testid="stRadio"] > label { display:none; }
    div[data-testid="stRadio"] div[role="radiogroup"] { gap:8px; background:#eeeff9; padding:6px; border-radius:14px; width:fit-content; }
    /* Keep every navigation label readable, including Streamlit's nested markdown elements. */
    div[data-testid="stRadio"] div[role="radiogroup"] label,
    div[data-testid="stRadio"] div[role="radiogroup"] label *,
    div[data-testid="stRadio"] div[role="radiogroup"] label p,
    div[data-testid="stRadio"] div[role="radiogroup"] label span { color:#26324d !important; opacity:1 !important; }
    div[data-testid="stRadio"] div[role="radiogroup"] label { background:transparent; border:1px solid transparent; border-radius:10px; padding:8px 15px; font-weight:800; cursor:pointer; transition:all .18s ease; }
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover { background:#dedbfb; border-color:#c8c0fb; }
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) { background:#7657e8; border-color:#7657e8; box-shadow:0 4px 12px #7657e82e; }
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked),
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) *,
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p,
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) span { color:#ffffff !important; opacity:1 !important; }
    .bottom-navigation { padding:18px 0 4px; border-top:1px solid #e7e9f4; margin-top:28px; }
    .bottom-navigation-title { color:#5140bd; font-size:12px; font-weight:800; letter-spacing:1px; margin-bottom:10px; }
    .bottom-navigation + div[data-testid="stHorizontalBlock"] .stButton > button,
    .bottom-navigation + div[data-testid="stHorizontalBlock"] .stButton > button p,
    .bottom-navigation + div[data-testid="stHorizontalBlock"] .stButton > button span { background:#f0edff; border:1px solid #d7cffd; color:#5038c7 !important; font-weight:800; opacity:1 !important; }
    .bottom-navigation + div[data-testid="stHorizontalBlock"] .stButton > button:hover { background:#7657e8; border-color:#7657e8; color:#ffffff !important; }
    .bottom-navigation + div[data-testid="stHorizontalBlock"] .stButton > button:disabled { background:#7657e8; border-color:#7657e8; color:#ffffff !important; opacity:1; }
    div[data-testid="stRadio"] div[role="radiogroup"] label p { font-size:12px; }
    .stButton > button[kind="primary"] { background:linear-gradient(110deg,#7657e8,#5a70dc); border:0; border-radius:11px; color:white; font-weight:800; min-height:43px; box-shadow:0 7px 16px #7657e82c; }
    .stButton > button[kind="secondary"] { border:1px solid #dedff0; border-radius:11px; color:#51447f; background:#fff; font-weight:700; min-height:42px; }
    [data-testid="stFileUploader"] { background:#fff; border:1.5px dashed #c9c6ee; border-radius:16px; padding:12px; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:16px; border-radius:14px; }
    [data-testid="stChatMessage"] { border:1px solid #ebeaf6; border-radius:15px; padding:12px 14px; }
    .soft-panel { background:#fff; border:1px solid var(--line); border-radius:18px; padding:22px; }
    /* High-contrast warning, code labels, and normal text */
    [data-testid="stAlert"] { background:#fff4d6 !important; border:1px solid #f0d58a !important; }
    [data-testid="stAlert"] p, [data-testid="stAlert"] div, [data-testid="stAlert"] span { color:#633f00 !important; opacity:1 !important; }
    .stMarkdown code { background:#eeeaff !important; color:#4930a8 !important; border:1px solid #d8cffd !important; }
    .stMarkdown p, .stMarkdown li { color:#45516a; opacity:1 !important; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color:#17213b !important; opacity:1 !important; }
    .footer { text-align:center; color:#8b92a8; font-size:11px; padding:26px 0 4px; }
    .bottom-navigation { padding:18px 0 4px; border-top:1px solid #e7e9f4; margin-top:28px; }
    .bottom-navigation-title { color:#5140bd !important; font-size:12px; font-weight:800; letter-spacing:1px; margin-bottom:10px; }
    div[data-testid="stHorizontalBlock"] .stButton > button { background:#f0edff !important; border:1px solid #d7cffd !important; color:#5038c7 !important; font-weight:800 !important; opacity:1 !important; }
    div[data-testid="stHorizontalBlock"] .stButton > button p, div[data-testid="stHorizontalBlock"] .stButton > button span { color:#5038c7 !important; opacity:1 !important; }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover { background:#7657e8 !important; border-color:#7657e8 !important; color:#fff !important; }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover p, div[data-testid="stHorizontalBlock"] .stButton > button:hover span { color:#fff !important; }
    div[data-testid="stHorizontalBlock"] .stButton > button:disabled { background:#7657e8 !important; border-color:#7657e8 !important; color:#fff !important; opacity:1 !important; }
    div[data-testid="stHorizontalBlock"] .stButton > button:disabled p, div[data-testid="stHorizontalBlock"] .stButton > button:disabled span { color:#fff !important; opacity:1 !important; }

    /* HIGH-CONTRAST DOCUMENT METRICS */
    [data-testid="stMetric"],
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] label p,
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricLabel"] p,
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricValue"] div {
        color: #17213b !important;
        opacity: 1 !important;
    }
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e3e7f2 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #53617b !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #17213b !important;
        font-weight: 800 !important;
    }

    /* VISIBLE, USABLE CHAT INPUT */
    [data-testid="stForm"] {
        background: #ffffff !important;
        border: 1px solid #dfe4f2 !important;
        border-radius: 14px !important;
        padding: 14px !important;
    }
    [data-testid="stForm"] label,
    [data-testid="stForm"] label p {
        color: #26324d !important;
        opacity: 1 !important;
    }
    [data-testid="stForm"] input,
    [data-testid="stTextInput"] input {
        background: #ffffff !important;
        color: #17213b !important;
        -webkit-text-fill-color: #17213b !important;
        caret-color: #7657e8 !important;
        border: 1px solid #cbd3e5 !important;
        opacity: 1 !important;
    }
    [data-testid="stForm"] input::placeholder,
    [data-testid="stTextInput"] input::placeholder {
        color: #68748c !important;
        -webkit-text-fill-color: #68748c !important;
        opacity: 1 !important;
    }
    [data-testid="stForm"] button {
        background: #7657e8 !important;
        border: 1px solid #7657e8 !important;
        color: #ffffff !important;
        opacity: 1 !important;
    }
    [data-testid="stForm"] button p,
    [data-testid="stForm"] button span {
        color: #ffffff !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        color: #26324d !important;
        opacity: 1 !important;
    }

    /* FINAL CONTRAST OVERRIDES: keep all important text readable */
    .hero h1, .hero h1 *, .stMarkdown .hero h1, .stMarkdown .hero h1 * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
        text-shadow: none !important;
    }
    .hero p, .stMarkdown .hero p {
        color: #f1f5ff !important;
        -webkit-text-fill-color: #f1f5ff !important;
        opacity: 1 !important;
    }
    .hero .eyebrow, .stMarkdown .hero .eyebrow {
        color: #a7fff0 !important;
        -webkit-text-fill-color: #a7fff0 !important;
        opacity: 1 !important;
    }
    .hero .hero-note, .stMarkdown .hero .hero-note {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
    }

    /* PDF uploader: light surface, dark labels, visible upload button */
    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        color: #26324d !important;
        border: 1.5px dashed #b8b1ee !important;
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] label *,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section *,
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] * {
        color: #26324d !important;
        -webkit-text-fill-color: #26324d !important;
        opacity: 1 !important;
    }
    [data-testid="stFileUploader"] section {
        background: #f7f8ff !important;
        border: 1px dashed #c9c6ee !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] button {
        background: #7657e8 !important;
        border: 1px solid #7657e8 !important;
        opacity: 1 !important;
    }
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] button * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Ensure metric values and labels are high contrast */
    [data-testid="stMetric"], [data-testid="stMetric"] *,
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #17213b !important;
        -webkit-text-fill-color: #17213b !important;
        opacity: 1 !important;
    }
    [data-testid="stMetric"] { background: #ffffff !important; }

    /* Make chat typing area and placeholder legible */
    [data-testid="stChatInput"], [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        background: #ffffff !important;
        color: #17213b !important;
        -webkit-text-fill-color: #17213b !important;
        caret-color: #7657e8 !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder {
        color: #68748c !important;
        -webkit-text-fill-color: #68748c !important;
        opacity: 1 !important;
    }

    @media(max-width:700px){.hero{padding:30px 23px}.brandbar{align-items:flex-start}.topbadge{display:none}.block-container{padding-left:1rem;padding-right:1rem}}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="brandbar">
      <div class="brandleft">
        <div class="brandicon">▤</div>
        <div><div class="brandname">DocuMind</div><div class="brandtag">DOCUMENT INTELLIGENCE</div></div>
      </div>
      <div class="topbadge">✦ AI-POWERED PDF ASSISTANT</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "page" not in st.session_state:
    st.session_state.page = "Overview"

def navigate_to(target):
    # Streamlit callbacks run before widgets are instantiated on the next rerun.
    st.session_state.page = target
if "messages" not in st.session_state:
    st.session_state.messages = []

page = st.radio(
    "Navigate",
    ["Overview", "Ask your PDF", "How it works"],
    horizontal=True,
    label_visibility="collapsed",
    key="page",
)

if page == "Overview":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">READ LESS. UNDERSTAND MORE.</div>
          <h1>Your documents.<br>Clearer answers.</h1>
          <p>Upload a PDF and ask questions in plain language. DocuMind searches the document for relevant passages, checks their relevance, and uses an AI model to generate an answer with source-page references.</p>
          <div class="hero-note">📄 PDF understanding&nbsp;&nbsp; · &nbsp;&nbsp;🔎 Semantic search&nbsp;&nbsp; · &nbsp;&nbsp;📚 Source pages</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-kicker">WHAT YOU CAN DO</div><div class="section-title">From long documents to useful answers</div><div class="section-copy">A retrieval-augmented generation workflow designed to keep responses grounded in the PDF you provide.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    cards = [
        ("📑", "Read PDF content", "Extract text from PDF pages and split it into manageable chunks for retrieval."),
        ("🔎", "Find relevant passages", "Create semantic embeddings and search for passages related to your question."),
        ("💬", "Answer with context", "Generate a response from retrieved content and show the source page numbers."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(f'<div class="feature-card"><div class="feature-icon">{icon}</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="section-kicker">THE PROCESS</div><div class="section-title">A clear, guided workflow</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    steps = [
        ("01", "Upload", "Choose the PDF you want to explore."),
        ("02", "Process", "Extract, chunk, embed, and index the document."),
        ("03", "Ask", "Ask questions and review answers with sources."),
    ]
    for col, (num, title, desc) in zip([p1, p2, p3], steps):
        with col:
            st.markdown(f'<div class="process-step"><div class="step-number">{num}</div><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)
    st.write("")
    st.info("Get started by selecting **Ask your PDF** in the navigation above.")
    st.markdown('<div class="footer">DocuMind · Document Intelligence RAG · Built with Streamlit, LangChain, FAISS and Hugging Face</div>', unsafe_allow_html=True)

elif page == "Ask your PDF":
    st.markdown('<div class="section-kicker">DOCUMENT WORKSPACE</div><div class="section-title">Ask questions about your PDF</div><div class="section-copy">Upload and process a document once, then ask questions grounded in its content.</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("#### 1. Add a document")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], help="Use a text-based PDF for best extraction results.")
        process_col, clear_col = st.columns([1, 1])
        with process_col:
            process_clicked = st.button("✦ Process PDF", type="primary", use_container_width=True, disabled=uploaded_file is None)
        with clear_col:
            clear_clicked = st.button("Clear current document", use_container_width=True)
        if clear_clicked:
            for key in ["vector_store", "llm", "pdf_name", "num_pages", "num_chunks", "messages"]:
                st.session_state.pop(key, None)
            st.session_state.messages = []
            st.rerun()
        if uploaded_file is not None:
            st.caption(f"Selected: **{uploaded_file.name}** · {uploaded_file.size / 1024:.1f} KB")
        if process_clicked and uploaded_file is not None:
            try:
                with st.spinner("Extracting text, building semantic search, and preparing the AI assistant…"):
                    documents = load_pdf(uploaded_file)
                    if not documents:
                        raise ValueError("No readable pages were found in this PDF.")
                    chunks = split_documents(documents)
                    if not chunks:
                        raise ValueError("No text chunks were extracted. Check whether the PDF contains selectable text.")
                    embedding_model = create_embedding_model()
                    vector_store = create_vector_store(chunks, embedding_model)
                    llm = create_llm()
                    st.session_state.vector_store = vector_store
                    st.session_state.llm = llm
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.num_pages = len(documents)
                    st.session_state.num_chunks = len(chunks)
                    st.session_state.messages = []
                st.success("Document processed successfully. You can start asking questions below.")
            except Exception as exc:
                st.error(f"Could not process this document: {exc}")
                st.caption("Check your environment variables, API key, installed dependencies, and PDF readability.")
    if "vector_store" in st.session_state:
        st.write("")
        st.markdown("#### 2. Document overview")
        m1, m2, m3 = st.columns(3)
        m1.metric("Document", st.session_state.pdf_name)
        m2.metric("Pages extracted", st.session_state.num_pages)
        m3.metric("Text chunks", st.session_state.num_chunks)
        st.write("")
        st.markdown("#### 3. Chat with your document")
        st.caption("Answers are generated from retrieved document context. If the document does not contain the answer, the assistant should say so.")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    st.caption("Sources: " + " · ".join(f"Page {p}" for p in message["sources"]))
        with st.form("chat_form", clear_on_submit=True):
            query = st.text_input(
                "Your question",
                placeholder="Ask a question about your PDF...",
                key="chat_query",
            )
            submitted = st.form_submit_button(
                "Send question",
                type="primary",
                use_container_width=True,
            )
        if submitted and query.strip():
            query = query.strip()
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                with st.spinner("Searching for relevant passages…"):
                    try:
                        results = retrieve_documents(st.session_state.vector_store, query, k=3)
                        if not results:
                            answer, relevant, pages = "I could not find the answer in the provided document.", False, []
                        else:
                            relevant = check_relevance(st.session_state.llm, query, results)
                            if not relevant:
                                answer, pages = "I could not find the answer in the provided document.", []
                            else:
                                context = "\n\n".join(result.page_content for result in results)
                                answer = generate_answer(st.session_state.llm, query, context)
                                pages = []
                                for result in results:
                                    page_label = result.metadata.get("page_label", result.metadata.get("page"))
                                    if page_label is not None and page_label not in pages:
                                        pages.append(page_label)
                        st.markdown(answer)
                        if relevant and pages:
                            st.caption("Sources: " + " · ".join(f"Page {p}" for p in pages))
                        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": pages if relevant else []})
                    except Exception as exc:
                        error_message = f"I couldn't complete the answer: {exc}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.rerun()
    else:
        st.markdown('<div class="soft-panel"><strong>Ready when you are.</strong><br><span style="color:#68748c;font-size:13px">After processing a PDF, the document statistics and chat workspace will appear here.</span></div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="section-kicker">BEHIND THE ANSWER</div><div class="section-title">How DocuMind works</div><div class="section-copy">The application uses retrieval-augmented generation (RAG) to find supporting passages before generating an answer.</div>', unsafe_allow_html=True)
    pipeline = [
        ("01", "PDF text extraction", "PyPDFLoader extracts page text and metadata from the uploaded document."),
        ("02", "Text chunking", "RecursiveCharacterTextSplitter creates chunks with overlap to preserve context across boundaries."),
        ("03", "Embeddings", "Hugging Face sentence-transformers/all-MiniLM-L6-v2 turns text chunks into vectors."),
        ("04", "Vector search", "FAISS stores the vectors and retrieves passages semantically related to the question."),
        ("05", "Relevance check", "The configured Groq language model checks whether retrieved content addresses the question."),
        ("06", "Answer + sources", "The model answers from retrieved context and the app displays source-page references when available."),
    ]
    for i in range(0, len(pipeline), 2):
        left, right = st.columns(2)
        for col, item in zip([left, right], pipeline[i:i+2]):
            num, title, desc = item
            with col:
                st.markdown(f'<div class="feature-card" style="margin-bottom:14px"><div class="step-number">{num}</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
    st.warning("Grounding is not a guarantee of correctness. Review the cited PDF pages for important decisions, and do not upload confidential documents unless your environment is approved for them.")
    st.markdown('<div class="section-title" style="font-size:20px;margin-top:22px">Technology stack</div>', unsafe_allow_html=True)
    st.markdown("`Python`　`Streamlit`　`LangChain`　`FAISS`　`Hugging Face Embeddings`　`Groq`　`PyPDF`")
    st.markdown('<div class="footer">DocuMind · Document Intelligence RAG</div>', unsafe_allow_html=True)


# Persistent page navigation at the bottom of every page.
st.markdown('<div class="bottom-navigation"><div class="bottom-navigation-title">EXPLORE DOCUMIND</div></div>', unsafe_allow_html=True)
bottom_cols = st.columns(3)
for col, target, label in zip(
    bottom_cols,
    ["Overview", "Ask your PDF", "How it works"],
    ["← Overview", "Ask your PDF →", "How it works →"],
):
    with col:
        st.button(
            label,
            key=f"bottom_nav_{target}",
            use_container_width=True,
            disabled=(page == target),
            on_click=navigate_to,
            args=(target,),
        )
st.markdown('<div class="footer">DocuMind · Document Intelligence</div>', unsafe_allow_html=True)

