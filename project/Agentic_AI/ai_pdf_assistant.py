import streamlit as st
from PyPDF2 import PdfReader
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.vectordb.chroma import ChromaDb
from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.models.groq import Groq
from agno.tools.googlesearch import GoogleSearchTools
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.chunking.fixed import FixedSizeChunking
from agno.vectordb.search import SearchType
from agno.knowledge.reranker.cohere import CohereReranker

import tempfile
import os
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Streamlit Page Setup ---
st.set_page_config(page_title="📘 PDF AI Assistant", page_icon="🤖", layout="wide")

# --- Title & Description ---
st.title("📘 PDF AI Assistant")
st.caption("Upload a PDF and ask questions — powered by Groq + Ollama embeddings")

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("🔍 Processing and embedding PDF..."):
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # --- Setup Embedder and Knowledge Base ---
        embedder = OllamaEmbedder(dimensions=1024, id="bge-m3")
        vectordb = ChromaDb(collection="pdf-reader", embedder=embedder,
                            search_type = SearchType.vector ,
                            reranker = CohereReranker())
        knowledge = Knowledge(vector_db=vectordb)

        # Add PDF content with chunking
        knowledge.add_content_async(
            path=tmp_path,
            reader=PDFReader(
                chunking_strategy=FixedSizeChunking(chunk_size=1000, overlap=100)
            ),
            metadata={"source": uploaded_file.name},
            skip_if_exists=True,
        )

        st.success("✅ PDF successfully embedded into the knowledge base!")

# --- Question Input ---
st.markdown("---")
question = st.text_input("💬 Ask a question about your uploaded PDF:")

if question:
    with st.spinner("🤔 Thinking..."):
        # --- Retrieve top chunks ---
        results = vectordb.search(question, limit=3)
        context = "\n\n".join([r.content for r in results])


        # --- Create AI Model & Agent ---
        groq_model = Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key)
        web_search_agent = Agent(
            name="pdf_ai_assistant",
            role="Answer questions from uploaded PDFs and search the web if no files given.",
            model=groq_model,
            tools=[GoogleSearchTools()],
            knowledge=knowledge,
            markdown=True,
            instructions=[
                "If a file is uploaded, use its content primarily and search on the web for related information.",
                "If no file is uploaded, use web search to answer the question.",
                "Summarize and explain clearly using markdown formatting.",
                "Use bullet points, headings, and short paragraphs for clarity."
            ],
        )

        # --- Run the agent ---
        response = web_search_agent.run(f"Context:\n{context}\n\nQuestion:\n{question}")

    # --- Display the result beautifully ---
    st.markdown("🧠 AI Response:")
    st.markdown(
        f"""
        <div style="background-color:#1e1e2f;padding:20px;border-radius:12px;margin-top:10px;">
            <p style="color:#dcdcdc;font-size:16px;line-height:1.6;">
                {response.content if hasattr(response, 'content') else response}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
