import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state
if 'constitution_processed' not in st.session_state:
    st.session_state.constitution_processed = False
if 'uploaded_files_processed' not in st.session_state:
    st.session_state.uploaded_files_processed = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = []

# Configuration - FREE TIER COMPATIBLE
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"  # Free alternative
CHROMA_DB_PATH = "./chroma_db"
DOCUMENT_SPLIT_CHUNK_SIZE = 600  # Smaller chunks for free tier
DOCUMENT_SPLIT_CHUNK_OVERLAP = 50

# Optimized prompt template
prompt_template = """<|system|>
You are a constitutional law expert. Answer strictly based on the context.
Keep responses under 2 sentences. If unsure, say "Not addressed in the Constitution."</s>
<|user|>
Context: {context}

Question: {question}</s>
<|assistant|>"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_and_process_constitution():
    try:
        with st.spinner("Downloading Constitution..."):
            loader = WebBaseLoader("https://www.akorda.kz/en/constitution-of-the-republic-of-kazakhstan-50912")
            documents = loader.load()

        with st.spinner("Processing text..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=DOCUMENT_SPLIT_CHUNK_SIZE,
                chunk_overlap=DOCUMENT_SPLIT_CHUNK_OVERLAP
            )
            texts = text_splitter.split_documents(documents)[:20]  # Strict limit

        with st.spinner("Building database..."):
            embeddings = get_embeddings()
            vector_store = Chroma.from_documents(
                texts,
                embeddings,
                persist_directory=CHROMA_DB_PATH
            )
            vector_store.persist()

        st.session_state.vector_store = vector_store
        st.session_state.constitution_processed = True
        return True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False


def process_uploaded_files(uploaded_files):
    if not uploaded_files:
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DOCUMENT_SPLIT_CHUNK_SIZE,
        chunk_overlap=DOCUMENT_SPLIT_CHUNK_OVERLAP
    )

    documents = []
    for uploaded_file in uploaded_files:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            with st.spinner(f"Reading {uploaded_file.name}..."):
                if file_ext == '.pdf':
                    loader = PyPDFLoader(tmp_file_path)
                elif file_ext == '.docx':
                    loader = Docx2txtLoader(tmp_file_path)
                elif file_ext == '.txt':
                    loader = TextLoader(tmp_file_path)
                else:
                    st.warning(f"Skipped {uploaded_file.name} (unsupported format)")
                    continue

                loaded_docs = loader.load()
                split_docs = text_splitter.split_documents(loaded_docs)[:3]  # Very strict limit
                documents.extend(split_docs)
                st.session_state.uploaded_docs.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    if documents:
        try:
            with st.spinner("Updating database..."):
                embeddings = get_embeddings()

                if st.session_state.vector_store:
                    st.session_state.vector_store.add_documents(documents)
                else:
                    st.session_state.vector_store = Chroma.from_documents(
                        documents,
                        embeddings,
                        persist_directory=CHROMA_DB_PATH
                    )

                st.session_state.vector_store.persist()
                st.session_state.uploaded_files_processed = True
                st.success(f"Added {len(documents)} sections")
        except Exception as e:
            st.error(f"Database error: {str(e)}")


def get_answer(question):
    if not st.session_state.vector_store:
        return "Please load the Constitution first."

    try:
        llm = HuggingFaceHub(
            repo_id=LLM_REPO_ID,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            model_kwargs={
                "temperature": 0.3,
                "max_new_tokens": 100,  # Strict limit
                "top_k": 10
            }
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 1}),  # Only 1 doc
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=False
        )

        result = qa_chain({"query": question})
        return result["result"]
    except Exception as e:
        return f"System busy. Please try a shorter question. ({str(e)})"


# Streamlit UI
st.set_page_config(
    page_title="🇰🇿 Constitution AI",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🇰🇿 Kazakhstan Constitution Assistant")
st.caption("Free tier compatible version")

# Sidebar
with st.sidebar:
    st.header("Documents")

    if st.button("🔄 Load Constitution", help="From official government website"):
        if load_and_process_constitution():
            st.rerun()

    uploaded_files = st.file_uploader(
        "📎 Add supporting files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Max 1MB per file recommended"
    )

    if st.button("⚙️ Process Files", disabled=not uploaded_files):
        process_uploaded_files(uploaded_files)
        st.rerun()

    if st.session_state.constitution_processed:
        st.success("Constitution loaded")
    if st.session_state.uploaded_docs:
        st.info(f"Extra docs: {len(st.session_state.uploaded_docs)}")

# Main Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_answer(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})