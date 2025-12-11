import streamlit as st
import os
from typing import List, Tuple
import numpy as np

# YouTube & Text Processing
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

# LangChain Core
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.documents import Document

# LangChain - Hugging Face
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

# BM25 for Hybrid Search
from rank_bm25 import BM25Okapi

# Sentence Transformers for Reranking
from sentence_transformers import CrossEncoder

from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables
load_dotenv()

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="🎥 YouTube Video Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)




st.markdown("""
<style>
    /* ============================================================ */
    /* MAIN LAYOUT & BACKGROUND */
    /* ============================================================ */
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
        padding: 0;
        overflow: hidden !important;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
        max-height: 100vh;
        overflow-y: auto;
        overflow-x: hidden;
    }
    
    /* Main content scrollbar */
    .main .block-container::-webkit-scrollbar {
        width: 10px;
    }
    
    .main .block-container::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    .main .block-container::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #ec4899 0%, #ef4444 100%);
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .main .block-container::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }

    /* ============================================================ */
    /* STREAMLIT HEADER BAR - CUSTOM COLOR */
    /* ============================================================ */
    
    /* Main header container */
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%) !important;
        border-bottom: 2px solid rgba(126, 34, 206, 0.5) !important;
    }
    
    /* Alternative: Solid color */
    header[data-testid="stHeader"] {
        background: #1e3c72 !important;
    }
    
    /* Style the toolbar (where Deploy button is) */
    header[data-testid="stHeader"] > div {
        background: transparent !important;
    }
    
    /* Style the Deploy button if you want */
    header[data-testid="stHeader"] button {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    header[data-testid="stHeader"] button:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        border-color: rgba(255, 255, 255, 0.5) !important;
    }
    
        
    /* ============================================================ */
    /* SIDEBAR STYLING */
    /* ============================================================ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%) !important;
        min-height: 100vh;
        overflow-y: auto !important;
        padding:10px;
    }
    
    /* Sidebar scrollbar */
    [data-testid="stSidebar"]::-webkit-scrollbar {
        width: 8px;
    }
    
    [data-testid="stSidebar"]::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 4px;
    }
    
    /* Clean transparent backgrounds for sidebar */
    [data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebar"] .element-container {
        background: transparent !important;
        margin-bottom: 0 !important;
        padding: 0 !important;
    }
    
    /* Hide empty containers */
    [data-testid="stSidebar"] .element-container:empty,
    [data-testid="stSidebar"] .element-container > div:empty {
        display: none !important;
    }
    
1
    [data-testid="stSidebar"] h2 {
        color: white !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        margin: 1rem 0 0.5rem 0 !important;
        padding: 0 !important;
    }
    
    /* Radio group container - LEFT ALIGN */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-start !important;
        width: 100% !important;
    }
    
    /* Radio button styling - BASE STYLES */
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background: transparent !important;
        padding: 0.7rem 1rem !important;
        border-radius: 8px !important;
        margin: 0.3rem 0 !important;
        color: white !important;
        border: none !important;
        transition: all 0.3s ease !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        white-space: nowrap !important;
        width: 100% !important;
        cursor: pointer !important;
    }
    
    /* Remove backgrounds from radio label children */
    [data-testid="stSidebar"] div[role="radiogroup"] label > div {
        border: none !important;
        box-shadow: none !important;
        display: inline-flex !important;
        flex-direction: row !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    
    [data-testid="stSidebar"] div[role="radiogroup"] label > div > div {
        background: transparent !important;
    }
    
    /* Fix text display */
    [data-testid="stSidebar"] div[role="radiogroup"] label p,
    [data-testid="stSidebar"] div[role="radiogroup"] label span {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        display: inline !important;
        white-space: nowrap !important;
    }
    
    /* Hover effect - FULL WIDTH */
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Selected radio button - ENHANCED VISIBILITY */
    [data-testid="stSidebar"] div[role="radiogroup"] label[aria-checked="true"] {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.35) 0%, rgba(239, 68, 68, 0.35) 100%) !important;
        font-weight: 700 !important;
        border-left: 5px solid #ec4899 !important;
        padding-left: 1rem !important;
        box-shadow: 0 2px 8px rgba(236, 72, 153, 0.3) !important;
        transform: translateX(3px) !important;
    }
    
    /* Add arrow indicator to selected navigation */
    [data-testid="stSidebar"] div[role="radiogroup"] label[aria-checked="true"]::before {
        content: "▶";
        margin-right: 0.5rem;
        font-size: 0.8rem;
        color: #ec4899;
    }
    
    /* Radio button circle - VISIBLE */
    [data-testid="stSidebar"] [data-baseweb="radio"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin-right: 0.5rem !important;
    }
    /* Checked radio button - white outer circle */
    [data-testid="stSidebar"] label[aria-checked="true"] [data-baseweb="radio"] > div:first-child {
        background-color: white !important;
        border-color: white !important;
    }
    
    /* Pink dot in center when checked - using ::after pseudo-element */
    [data-testid="stSidebar"] label[aria-checked="true"] [data-baseweb="radio"] > div:first-child::after {
        content: '' !important;
        display: block !important;
        width: 8px !important;
        height: 8px !important;
        border-radius: 50% !important;
        background-color: #ec4899 !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
    }
    
    /* ============================================================ */
    /* MAIN CONTENT AREA - WITH PURPLE GLOW */
    /* ============================================================ */


    /* Title container with MASSIVE purple glow */
    .title-container {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
        padding: 1.5rem 2rem;
        border-radius: 20px;
        box-shadow: 
            0 0 80px 15px rgba(126, 34, 206, 0.6),
            0 0 120px 25px rgba(126, 34, 206, 0.4),
            0 10px 40px rgba(126, 34, 206, 0.5),
            0 4px 20px rgba(0, 0, 0, 0.2) !important;
        margin: 5rem 0 1.5rem 0;
        border: 3px solid rgba(126, 34, 206, 0.6);
    }
    
    .title-container h1 {
        color: #1e3c72 !important;
        margin: 0;
        font-size: 2rem;
    }
    
    .title-container p {
        color: #64748b !important;
        margin: 0.3rem 0 0 0;
}
    /* ============================================================ */
    /* VIDEO INFO HEADER - WITH PURPLE GLOW */
    /* ============================================================ */
    /* Video info header with MASSIVE purple glow */
    .video-info-header {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%) !important;
        padding: 1rem 1.5rem !important;
        border-radius: 15px !important;
        margin-bottom: 1rem !important;
        box-shadow: 
            0 0 70px 12px rgba(126, 34, 206, 0.6),
            0 0 100px 20px rgba(126, 34, 206, 0.4),
            0 8px 32px rgba(126, 34, 206, 0.5),
            0 4px 16px rgba(0, 0, 0, 0.2) !important;
        border: 3px solid rgba(126, 34, 206, 0.6) !important;
    }
    
    .video-info-header h3 {
        color: #1e3c72 !important;
        text-shadow: none !important;
        margin: 0 !important;
        font-size: 1.3rem !important;
        border-bottom: none !important;
        padding-bottom: 0 !important;
    }
    
    .video-info-header p {
        color: #475569 !important;
        margin: 0.5rem 0 0 0 !important;
        font-size: 0.95rem !important;
    }
    
    .video-info-header strong {
        color: #1e3c72 !important;
    }
    
    .video-info-header code {
        background: rgba(126, 34, 206, 0.15) !important;
        color: #7e22ce !important;
        padding: 0.2rem 0.6rem !important;
        border-radius: 6px !important;
        font-weight: bold !important;
    }

    
    /* ============================================================ */
    /* COLUMN CONTAINERS - WITH SUBTLE PURPLE GLOW */
    /* ============================================================ */
    /* Add glow to video and chat columns */
    [data-testid="column"] > div:first-child {
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(126, 34, 206, 0.2),
                    0 0 30px rgba(126, 34, 206, 0.1);
    }

    /* ============================================================ */
    /* COLUMN HEADERS & VIDEO PLAYER */
    /* ============================================================ */
    /* Column headers - Video Player and Chat */
    [data-testid="column"] h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-weight: bold;
        font-size: 1.4rem !important;
        background: transparent !important;
        padding: 0 !important;
        margin-bottom: 1rem !important;
        margin-top: 0 !important;
        border-bottom: 3px solid rgba(255, 255, 255, 0.3);
        padding-bottom: 0.5rem !important;
    }
    
    /* Full width columns */
    [data-testid="column"] {
        padding: 0 0.5rem;
    }
    
    /* Video iframe */
    iframe {
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        height: 500px !important;
        width: 100% !important;
    }
    
    /* ============================================================ */
    /* EXPANDERS & INFO BOXES - ENHANCED SHADOWS */
    /* ============================================================ */
    /* Expander styling with glow */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px;
        border: 2px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(126, 34, 206, 0.2);
    }
    
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] strong {
        color: white !important;
    }
    /* Expander inside chat messages - DARK TEXT */
    [data-testid="stChatMessage"] [data-testid="stExpander"] {
        background: rgba(126, 34, 206, 0.1) !important;
        border: 2px solid rgba(126, 34, 206, 0.3) !important;
    }
    
    [data-testid="stChatMessage"] [data-testid="stExpander"] summary,
    [data-testid="stChatMessage"] [data-testid="stExpander"] p,
    [data-testid="stChatMessage"] [data-testid="stExpander"] strong {
        color: #1e293b !important;
    }
    
    /* SHARED STYLES for info-box and tech-card */
    .info-box,
    .tech-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%) !important;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    /* Info boxes with MASSIVE purple glow */
    .info-box {
        padding: 1.5rem !important;
        border-radius: 15px !important;
        border-left: 5px solid #7e22ce !important;
        box-shadow: 
            0 0 70px 12px rgba(126, 34, 206, 0.6),
            0 0 100px 20px rgba(126, 34, 206, 0.4),
            0 8px 32px rgba(126, 34, 206, 0.5),
            0 4px 16px rgba(0, 0, 0, 0.2) !important;
        border: 3px solid rgba(126, 34, 206, 0.5) !important;
    }
    
    /* Tech cards with MASSIVE purple glow */
    .tech-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%) !important;
        padding: 2rem !important;
        border-radius: 20px !important;
        border-left: 6px solid #7e22ce !important;
        box-shadow: 
            0 0 80px 15px rgba(126, 34, 206, 0.6),
            0 0 120px 25px rgba(126, 34, 206, 0.4),
            0 12px 48px rgba(126, 34, 206, 0.5),
            0 4px 24px rgba(0, 0, 0, 0.2) !important;
        border: 3px solid rgba(126, 34, 206, 0.5) !important;
    }
    
    /* Headers in tech cards - PURPLE */
    .tech-card h3, 
    .tech-card h4 {
        color: #7e22ce !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.8rem !important;
        font-weight: 600 !important;
    }
    
    .tech-card h3 {
        font-size: 1.5rem !important;
        margin-top: 0 !important;
    }
    
    .tech-card h4 {
        font-size: 1.2rem !important;
    }
    
    /* Paragraphs and text in tech cards */
    .tech-card p {
        color: #1e293b !important;
        line-height: 1.7 !important;
        margin: 0.8rem 0 !important;
    }
    
    /* Lists in tech cards */
    .tech-card ul,
    .tech-card ol {
        color: #1e293b !important;
        margin: 1rem 0 !important;
        padding-left: 1.5rem !important;
    }
    
    .tech-card li {
        color: #1e293b !important;
        margin: 0.5rem 0 !important;
        line-height: 1.6 !important;
    }
    
    .tech-card li strong {
        color: #7e22ce !important;
        font-weight: 600 !important;
    }

    /* ============================================================ */
    /* CHAT AREA */
    /* ============================================================ */
    
    /* Chat scrollbar */
    [data-testid="stVerticalBlock"]:has([data-testid="stChatMessage"])::-webkit-scrollbar {
        width: 10px;
    }
    
    [data-testid="stVerticalBlock"]:has([data-testid="stChatMessage"])::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    [data-testid="stVerticalBlock"]:has([data-testid="stChatMessage"])::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #ec4899 0%, #ef4444 100%);
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    [data-testid="stVerticalBlock"]:has([data-testid="stChatMessage"])::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    
    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 2px solid rgba(126, 34, 206, 0.2);
        color: #1e293b !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* User message styling */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: linear-gradient(135deg, #ec4899 0%, #ef4444 100%);
        color: white !important;
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] div {
        color: inherit !important;
    }
    

    /* Code styling */
/* ============================================================ */
/* CODE BLOCKS - COMPLETE STYLING */
/* ============================================================ */

    /* Pre blocks (code fences) in tech cards and info boxes */
    .tech-card pre,
    .info-box pre {
        background: #1e293b !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        overflow-x: auto !important;
        border: 2px solid rgba(126, 34, 206, 0.3) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Code inside pre blocks - light colored text */
    .tech-card pre code,
    .info-box pre code {
        background: transparent !important;
        color: #e2e8f0 !important;
        font-size: 0.9rem !important;
        line-height: 1.6 !important;
        font-family: 'Courier New', monospace !important;
        padding: 0 !important;
    }
    
    /* Inline code in light backgrounds (not in pre blocks) */
    .tech-card code:not(pre code),
    .info-box code:not(pre code),
    .video-info-header code {
        background: rgba(126, 34, 206, 0.15) !important;
        color: #7e22ce !important;
        padding: 0.2rem 0.6rem !important;
        border-radius: 6px !important;
        font-weight: bold !important;
        font-size: 0.9em !important;
    }
    
    /* General inline code (fallback) */
    code {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #fbbf24 !important;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.9em;
    }
    
    /* Scrollbar for code blocks */
    .tech-card pre::-webkit-scrollbar,
    .info-box pre::-webkit-scrollbar {
        height: 8px;
    }
    
    .tech-card pre::-webkit-scrollbar-track,
    .info-box pre::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 4px;
    }
    
    .tech-card pre::-webkit-scrollbar-thumb,
    .info-box pre::-webkit-scrollbar-thumb {
        background: rgba(126, 34, 206, 0.5);
        border-radius: 4px;
    }
    
    .tech-card pre::-webkit-scrollbar-thumb:hover,
    .info-box pre::-webkit-scrollbar-thumb:hover {
        background: rgba(126, 34, 206, 0.7);
    }
    
    /* ============================================================ */
    /* BUTTONS */
    /* ============================================================ */
    /* Primary buttons (example videos) */
    button[kind="primary"] {
        background: linear-gradient(135deg, #7e22ce 0%, #ec4899 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 1.5rem !important;
        font-weight: 600 !important;
        margin-top: 1rem !important;
        box-shadow: 0 4px 8px rgba(126, 34, 206, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #6b21a8 0%, #db2777 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(126, 34, 206, 0.4) !important;
    }
    
    /* Secondary buttons */
    button[kind="secondary"] {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%) !important;
        color: #1e3c72 !important;
        border: 2px solid rgba(126, 34, 206, 0.3) !important;
        padding: 1rem !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    
    button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%) !important;
        border-color: #7e22ce !important;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    """Centralized configuration"""
    # Try Streamlit secrets first (for cloud), then .env file (for local)
    try:
        HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
    except:
        HF_TOKEN = os.getenv("HF_TOKEN", "")
    
    if not HF_TOKEN:
        st.error("⚠️ HuggingFace token not found! Please add HF_TOKEN to your .env file.")
        st.info("Get your free token at: https://huggingface.co/settings/tokens")
        st.stop()
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 10
    FINAL_K = 4
    ALPHA = 0.5
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MAX_HISTORY = 5

os.environ["HUGGING_FACE_ACCESS_TOKEN"] = Config.HF_TOKEN

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def fetch_youtube_transcript(video_id: str) -> Tuple[str, dict]:
    """Fetch transcript from YouTube video"""
    try:
        fetched_transcript = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        transcript_obj = fetched_transcript.to_raw_data()
        transcript = " ".join([entry["text"] for entry in transcript_obj])
        
        metadata = {
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "length": len(transcript),
            "num_segments": len(transcript_obj)
        }
        return transcript, metadata
    except Exception as e:
        raise Exception(f"Error fetching transcript: {e}")

def chunk_transcript(transcript: str, metadata: dict) -> List[Document]:
    """Split transcript into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.create_documents(texts=[transcript], metadatas=[metadata])
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks

def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    if "youtube.com/watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    else:
        return url

# ============================================================
# HYBRID RETRIEVER
# ============================================================
class HybridRetriever:
    """Hybrid retrieval combining FAISS and BM25 with reranking"""
    
    def __init__(self, vector_store, bm25_index, chunks, reranker, k=10, final_k=4, alpha=0.5):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.chunks = chunks
        self.reranker = reranker
        self.k = k
        self.final_k = final_k
        self.alpha = alpha
    
    def retrieve(self, query: str) -> List[Document]:
        """Execute hybrid retrieval with reranking"""
        # Semantic Search
        semantic_results = self.vector_store.similarity_search_with_score(query, k=self.k)
        
        # Keyword Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[-self.k:][::-1]
        
        # Combine Results
        combined_scores = {}
        semantic_scores = [score for _, score in semantic_results]
        max_semantic = max(semantic_scores) if semantic_scores else 1
        
        for doc, score in semantic_results:
            doc_id = id(doc)
            normalized_score = score / max_semantic
            combined_scores[doc_id] = {'doc': doc, 'score': self.alpha * normalized_score}
        
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        for idx in bm25_top_indices:
            doc = self.chunks[idx]
            doc_id = id(doc)
            normalized_score = bm25_scores[idx] / max_bm25
            if doc_id in combined_scores:
                combined_scores[doc_id]['score'] += (1 - self.alpha) * normalized_score
            else:
                combined_scores[doc_id] = {'doc': doc, 'score': (1 - self.alpha) * normalized_score}
        
        sorted_results = sorted(combined_scores.values(), key=lambda x: x['score'], reverse=True)[:self.k]
        
        # Reranking
        candidates = [item['doc'] for item in sorted_results]
        pairs = [[query, doc.page_content] for doc in candidates]
        rerank_scores = self.reranker.predict(pairs)
        reranked_indices = np.argsort(rerank_scores)[-self.final_k:][::-1]
        return [candidates[i] for i in reranked_indices]

# ============================================================
# MULTI-QUERY RETRIEVER
# ============================================================
class MultiQueryRetriever:
    """Multi-query retrieval for better coverage"""
    
    def __init__(self, base_retriever, llm, num_queries=3):
        self.base_retriever = base_retriever
        self.llm = llm
        self.num_queries = num_queries
        self.multi_query_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant that generates multiple search queries.

Given a user question, generate 3 different versions of the question to retrieve relevant documents.
Provide these alternative questions separated by newlines.

Original question: {question}

Alternative questions:
""")
    
    def generate_queries(self, question: str) -> List[str]:
        """Generate multiple query variations"""
        try:
            prompt = self.multi_query_prompt.invoke({"question": question})
            response = self.llm.invoke(prompt)
            queries = [q.strip() for q in response.content.split('\n') if q.strip()]
            return [question] + queries[:self.num_queries-1]
        except:
            return [question]
    
    def retrieve(self, question: str) -> Tuple[List[Document], List[str]]:
        """Retrieve using multiple query variations"""
        queries = self.generate_queries(question)
        all_docs = []
        seen_content = set()
        
        for query in queries:
            docs = self.base_retriever.retrieve(query)
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)
        
        if len(all_docs) > Config.FINAL_K:
            pairs = [[question, doc.page_content] for doc in all_docs]
            rerank_scores = self.base_retriever.reranker.predict(pairs)
            reranked_indices = np.argsort(rerank_scores)[-Config.FINAL_K:][::-1]
            final_docs = [all_docs[i] for i in reranked_indices]
        else:
            final_docs = all_docs
        
        return final_docs, queries

# ============================================================
# RAG CHAIN
# ============================================================
def format_docs(docs: List[Document]) -> str:
    """Format documents for context"""
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

def create_rag_chain(retriever, llm):
    """Create the main RAG chain"""
    rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant answering questions about a YouTube video.

{conversation_history}

Use the following context from the video to answer the question.
If the context doesn't contain the answer, say you don't know.
Be concise but informative.

Context from video:
{context}

Current question: {question}

Answer:
""")
    
    def retrieve_with_queries(q):
        docs, queries = retriever.retrieve(q)
        return {"docs": docs, "queries": queries}
    
    retrieval_runnable = RunnableLambda(retrieve_with_queries)
    memory_runnable = RunnableLambda(lambda _: st.session_state.get('conversation_history', 'No previous conversation.'))
    
    chain = (
        RunnableParallel({
            "question": RunnablePassthrough(),
            "retrieval_result": retrieval_runnable,
            "conversation_history": memory_runnable
        })
        .assign(
            contexts=lambda x: x["retrieval_result"]["docs"],
            queries=lambda x: x["retrieval_result"]["queries"],
            context=lambda x: format_docs(x["retrieval_result"]["docs"])
        )
        .assign(answer=rag_prompt | llm | StrOutputParser())
    )
    
    return chain

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
def init_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = ""
    if 'video_metadata' not in st.session_state:
        st.session_state.video_metadata = None
    if 'total_questions' not in st.session_state:
        st.session_state.total_questions = 0
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Chat"
    if 'pending_question' not in st.session_state:
        st.session_state.pending_question = None
# ============================================================
# SETUP CHATBOT
# ============================================================
@st.cache_resource
def setup_chatbot(video_id: str):
    """Setup chatbot with caching"""
    with st.spinner("🚀 Setting up chatbot..."):
        # Fetch transcript
        transcript, metadata = fetch_youtube_transcript(video_id)
        
        # Chunk transcript
        chunks = chunk_transcript(transcript, metadata)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Create BM25 index
        tokenized_docs = [doc.page_content.lower().split() for doc in chunks]
        bm25_index = BM25Okapi(tokenized_docs)
        
        # Load reranker
        reranker = CrossEncoder(Config.RERANKER_MODEL)
        
        # Initialize LLM
        llm = HuggingFaceEndpoint(
            repo_id=Config.LLM_MODEL,
            huggingfacehub_api_token=Config.HF_TOKEN,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95
        )
        chat_model = ChatHuggingFace(llm=llm)
        
        # Create retrievers
        hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            chunks=chunks,
            reranker=reranker
        )
        
        retriever = MultiQueryRetriever(
            base_retriever=hybrid_retriever,
            llm=chat_model
        )
        
        # Create chain
        chain = create_rag_chain(retriever=retriever, llm=chat_model)
        
        return chain, metadata, len(chunks)

# ============================================================
# TECH STACK PAGE
# ============================================================
def show_tech_stack():
    """Display technical details and architecture"""
    st.markdown("""
    <div class="title-container">
        <h1 style="text-align: center; color: #667eea; margin: 0;">
            🛠️ Technical Architecture
        </h1>
        <p style="text-align: center; color: #666; margin-top: 0.5rem;">
            Advanced RAG system with hybrid retrieval and multi-query generation
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Core Technologies
    st.markdown("""
    <div class="tech-card">
        <h3>🎯 Core Technologies</h3>
        <ul>
            <li><strong>LangChain</strong> - Framework for building LLM applications</li>
            <li><strong>Streamlit</strong> - Web application framework</li>
            <li><strong>HuggingFace</strong> - LLM inference and embeddings</li>
            <li><strong>FAISS</strong> - Facebook AI Similarity Search for vector storage</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Retrieval System - ALL IN ONE BOX
    st.markdown("""
    <div class="tech-card">
        <h3>🔍 Advanced Retrieval System</h3>
        <h4>1. Hybrid Retrieval</h4>
        <ul>
            <li><strong>Semantic Search (FAISS)</strong> - Finds contextually similar content using embeddings</li>
            <li><strong>Keyword Search (BM25)</strong> - Traditional information retrieval for exact matches</li>
            <li><strong>Score Fusion</strong> - Combines both approaches with configurable alpha (0.5)</li>
        </ul>
        <h4>2. Multi-Query Generation</h4>
        <ul>
            <li>Generates 3 variations of user questions</li>
            <li>Increases retrieval coverage and recall</li>
            <li>Captures different aspects of the query</li>
        </ul>
        <h4>3. Cross-Encoder Reranking</h4>
        <ul>
            <li><strong>Model:</strong> ms-marco-MiniLM-L-6-v2</li>
            <li>Reranks retrieved documents for relevance</li>
            <li>Ensures top-k results are most relevant</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Details
    st.markdown("""
    <div class="tech-card">
        <h3>🤖 Model Architecture</h3>
        <ul>
            <li><strong>LLM:</strong> Mistral-7B-Instruct-v0.2 (7 billion parameters)</li>
            <li><strong>Embeddings:</strong> BAAI/bge-base-en-v1.5 (768 dimensions)</li>
            <li><strong>Reranker:</strong> Cross-Encoder/ms-marco-MiniLM-L-6-v2</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Processing Pipeline
    st.markdown("""
    <div class="tech-card">
        <h3>⚙️ Processing Pipeline</h3>
        <ol>
            <li><strong>Transcript Extraction</strong> - Fetch YouTube transcript via API</li>
            <li><strong>Text Chunking</strong> - Split into 1000-char chunks with 200-char overlap</li>
            <li><strong>Embedding Generation</strong> - Create vector representations</li>
            <li><strong>Index Creation</strong> - Build FAISS vector store and BM25 index</li>
            <li><strong>Query Processing</strong> - Multi-query generation → Hybrid retrieval → Reranking</li>
            <li><strong>Response Generation</strong> - Context-aware LLM generation</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features
    st.markdown("""
    <div class="tech-card">
        <h3>✨ Key Features</h3>
        <ul>
            <li>💬 <strong>Conversation Memory</strong> - Maintains chat history for context-aware responses</li>
            <li>🔄 <strong>Dynamic Retrieval</strong> - Adapts to query complexity</li>
            <li>📊 <strong>Performance Optimization</strong> - Caching with @st.cache_resource</li>
            <li>🎨 <strong>Modern UI</strong> - Gradient design with responsive layout</li>
            <li>🎥 <strong>Embedded Video Player</strong> - Watch while you chat</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="tech-card">
            <h3>⚙️ Retrieval Config</h3>
            <ul>
                <li>Chunk Size: 1000</li>
                <li>Chunk Overlap: 200</li>
                <li>Retrieval K: 10</li>
                <li>Final K: 4</li>
                <li>Alpha (fusion): 0.5</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tech-card">
            <h3>🎛️ LLM Config</h3>
            <ul>
                <li>Max Tokens: 512</li>
                <li>Temperature: 0.7</li>
                <li>Top P: 0.95</li>
                <li>Max History: 5</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Use Cases
    st.markdown("""
    <div class="tech-card">
        <h3>🎯 Use Cases & Benefits</h3>
        <ul>
            <li>📚 <strong>Educational Content</strong> - Quick reference for lectures and tutorials</li>
            <li>🎓 <strong>Research</strong> - Extract insights from long-form video content</li>
            <li>💼 <strong>Professional Development</strong> - Efficient learning from conference talks</li>
            <li>🔍 <strong>Content Discovery</strong> - Find specific information without watching entire videos</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


                
def show_chat_page():
    """Display main chat interface"""
    # Header
    st.markdown("""
    <div class="title-container">
        <h1 style="text-align: center; color: #667eea; margin: 0;">
            🎥 Advanced RAG Based YouTube Video Chatbot
        </h1>
        <p style="text-align: center; color: #666; margin-top: 0.5rem;">
            Ask questions about any YouTube video with AI-powered retrieval
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        video_input = st.text_input(
            "YouTube Video URL or ID",
            placeholder="https://www.youtube.com/watch?v=...",
            key="video_input"
        )
        
        if st.button("🚀 Initialize Chatbot", use_container_width=True):
            if video_input:
                try:
                    video_id = extract_video_id(video_input)
                    chain, metadata, num_chunks = setup_chatbot(video_id)
                    
                    st.session_state.chain = chain
                    st.session_state.video_metadata = metadata
                    st.session_state.num_chunks = num_chunks
                    st.session_state.video_id = video_id
                    st.session_state.initialized = True
                    st.session_state.messages = []
                    st.session_state.conversation_history = ""
                    st.session_state.total_questions = 0
                    st.session_state.pending_question = None
                    
                    st.success("✅ Chatbot initialized!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.warning("⚠️ Please enter a video URL or ID")
        
        st.markdown("---")

        if st.session_state.initialized:
            st.markdown("### 📊 Video Info")
            st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.1); 
                            padding: 1rem; border-radius: 10px; 
                            border: 2px solid rgba(255, 255, 255, 0.2);">
                    <p style="margin: 0.3rem 0;"><strong>Video ID:</strong><br>{st.session_state.video_metadata['video_id']}</p>
                    <p style="margin: 0.3rem 0;"><strong>Chunks:</strong> {st.session_state.num_chunks}</p>
                    <p style="margin: 0.3rem 0;"><strong>Questions:</strong> {st.session_state.total_questions}</p>
                </div>
            """, unsafe_allow_html=True)
        
            
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_history = ""
                st.session_state.total_questions = 0
                st.session_state.pending_question = None
                st.rerun()
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Tips
        - Ask specific questions about the video
        - Follow-up questions use conversation history
        - Clear chat to start fresh
        - Watch the video while chatting!
        """)
    
    # Main chat interface
    if not st.session_state.initialized:
        st.markdown("""
        <div class="info-box" style="text-align: center; padding: 3rem;">
            <h3 style="color:black;">👋 Welcome!</h3>
            <p style="color:black;">Enter a YouTube video URL in the sidebar to get started.</p>
            <p style="color:brown;">The chatbot will analyze the video transcript and answer your questions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example videos with better styling
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%); 
                    padding: 2rem; border-radius: 20px; margin: 2rem 0;
                    box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
            <h3 style="color: #1e3c72; text-align: center; margin-bottom: 1.5rem;">
                🎬 Try These Example Videos
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 15px; 
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center;
                        border: 2px solid rgba(126, 34, 206, 0.2);">
                <h4 style="color: #7e22ce; margin-bottom: 1rem;">📚 Attention Mechanism</h4>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Load Video", use_container_width=True, key="ex1", type="primary"):
                video_id = "kCc8FmEb1nY"
                try:
                    chain, metadata, num_chunks = setup_chatbot(video_id)
                    st.session_state.chain = chain
                    st.session_state.video_metadata = metadata
                    st.session_state.num_chunks = num_chunks
                    st.session_state.video_id = video_id
                    st.session_state.initialized = True
                    st.session_state.messages = []
                    st.session_state.conversation_history = ""
                    st.session_state.total_questions = 0
                    st.session_state.pending_question = None
                    st.success("✅ Chatbot initialized!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        with col2:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 15px; 
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center;
                        border: 2px solid rgba(126, 34, 206, 0.2);">
                <h4 style="color: #7e22ce; margin-bottom: 1rem;">🧠 Neural Networks</h4>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Load Video", use_container_width=True, key="ex2", type="primary"):
                video_id = "aircAruvnKk"
                try:
                    chain, metadata, num_chunks = setup_chatbot(video_id)
                    st.session_state.chain = chain
                    st.session_state.video_metadata = metadata
                    st.session_state.num_chunks = num_chunks
                    st.session_state.video_id = video_id
                    st.session_state.initialized = True
                    st.session_state.messages = []
                    st.session_state.conversation_history = ""
                    st.session_state.total_questions = 0
                    st.session_state.pending_question = None
                    st.success("✅ Chatbot initialized!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        with col3:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 15px; 
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center;
                        border: 2px solid rgba(126, 34, 206, 0.2);">
                <h4 style="color: #7e22ce; margin-bottom: 1rem;">🤖 Transformers</h4>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Load Video", use_container_width=True, key="ex3", type="primary"):
                video_id = "zxQyTK8quyY"
                try:
                    chain, metadata, num_chunks = setup_chatbot(video_id)
                    st.session_state.chain = chain
                    st.session_state.video_metadata = metadata
                    st.session_state.num_chunks = num_chunks
                    st.session_state.video_id = video_id
                    st.session_state.initialized = True
                    st.session_state.messages = []
                    st.session_state.conversation_history = ""
                    st.session_state.total_questions = 0
                    st.session_state.pending_question = None
                    st.success("✅ Chatbot initialized!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    else:
        
        # Add header section with video title
        st.markdown(f"""
            <div class="video-info-header">
                <h3>🎬 Now Chatting About Video</h3>
                <p>
                    <strong>Video ID:</strong> <code>{st.session_state.video_id}</code>
                    <span style="margin: 0 0.5rem;">•</span>
                    <strong>{st.session_state.num_chunks}</strong> chunks
                    <span style="margin: 0 0.5rem;">•</span>
                    <strong>{st.session_state.total_questions}</strong> questions
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Create two columns - video on left, chat on right
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Embed YouTube video
            st.markdown("### 🔹 Video Player")
            
            video_url = f"https://www.youtube.com/embed/{st.session_state.video_id}?enablejsapi=1&rel=0"
            st.markdown(f"""
                <div id="video-wrapper">
                    <iframe 
                        id="youtube-player"
                        width="100%" 
                        height="500" 
                        src="{video_url}" 
                        frameborder="0" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen>
                    </iframe>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 💬 Chat")
            
            # Display chat messages in container
            chat_container = st.container(height=500)
            
            with chat_container:
                # Display all messages
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        with st.chat_message("user"):
                            st.write(message["content"])
                    else:
                        with st.chat_message("assistant"):
                            st.write(message["content"])
                            if "queries" in message and len(message["queries"]) > 1:
                                with st.expander("🔍 View Generated Query Variations"):
                                    for idx, q in enumerate(message["queries"], 1):
                                        clean_query = q.strip()
                                        import re
                                        clean_query = re.sub(r'^\d+\.\s*', '', clean_query)
                                        st.markdown(f"**Query {idx}:** {clean_query}")

                

                user_question = st.chat_input("Ask a question about the video...")
                
                # Process question immediately when entered
                if user_question:
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": user_question})
                    
                    # Show thinking indicator
                    with chat_container:
                        with st.chat_message("assistant"):
                            with st.spinner("🤔 Analyzing video transcript..."):
                                try:
                                    result = st.session_state.chain.invoke(user_question)
                                    answer = result["answer"].strip()
                                    queries = result["queries"]
                                    
                                    # Update conversation history
                                    st.session_state.conversation_history += f"Q: {user_question}\nA: {answer}\n\n"
                                    
                                    # Keep only last MAX_HISTORY conversations
                                    history_lines = st.session_state.conversation_history.split('\n\n')
                                    if len(history_lines) > Config.MAX_HISTORY * 2:
                                        st.session_state.conversation_history = '\n\n'.join(history_lines[-(Config.MAX_HISTORY * 2):])
                                    
                                    st.session_state.total_questions += 1
                                    
                                    # Add bot message
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": answer,
                                        "queries": queries
                                    })
                                    
                                except Exception as e:
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": f"❌ I apologize, but I encountered an error: {str(e)}\n\nPlease try again."
                                    })
                    
                    st.rerun()
        

        

        
# ============================================================
# ABOUT PAGE
# ============================================================
def show_about_page():
    """Display about information"""
    st.markdown("""
    <div class="title-container">
        <h1 style="text-align: center; color: #667eea; margin: 0;">
            ℹ️ About This Project
        </h1>
        <p style="text-align: center; color: #666; margin-top: 0.5rem;">
            Learn more about the YouTube Video Chatbot
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tech-card">
        <h3>🎯 Project Overview</h3>
        <p>
            The YouTube Video Chatbot is an advanced AI-powered application that allows users to have 
            natural conversations about YouTube video content. Instead of watching entire videos, users 
            can ask specific questions and get accurate, context-aware answers extracted directly from 
            the video transcript.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tech-card">
        <h3>🚀 Key Capabilities</h3>
        <ul>
            <li><strong>Instant Q&A:</strong> Ask questions and get immediate answers from video content</li>
            <li><strong>Context-Aware:</strong> Maintains conversation history for follow-up questions</li>
            <li><strong>Multi-Query Search:</strong> Generates query variations for comprehensive retrieval</li>
            <li><strong>Hybrid Retrieval:</strong> Combines semantic and keyword search for best results</li>
            <li><strong>Smart Reranking:</strong> Uses cross-encoder models to prioritize most relevant content</li>
            <li><strong>Embedded Player:</strong> Watch the video while chatting for better context</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tech-card">
        <h3>💡 How It Works</h3>
        <ol>
            <li><strong>Enter Video URL:</strong> Paste any YouTube video URL or ID</li>
            <li><strong>Automatic Processing:</strong> The system fetches and processes the transcript</li>
            <li><strong>Ask Questions:</strong> Type your questions in natural language</li>
            <li><strong>Get Answers:</strong> Receive accurate, context-aware responses instantly</li>
            <li><strong>Continue Conversation:</strong> Ask follow-up questions with full context</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="tech-card">
            <h3>✅ Best Practices</h3>
            <ul>
                <li>Ask specific, focused questions</li>
                <li>Use the video player for reference</li>
                <li>Build on previous questions naturally</li>
                <li>Clear chat to start fresh topics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tech-card">
            <h3>⚠️ Limitations</h3>
            <ul>
                <li>Requires English transcripts</li>
                <li>Video must have captions enabled</li>
                <li>Response quality depends on transcript accuracy</li>
                <li>Processing time varies with video length</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tech-card">
        <h3>🌟 Future Enhancements</h3>
        <ul>
            <li>Support for multiple languages</li>
            <li>Video timestamp references in answers</li>
            <li>Ability to compare multiple videos</li>
            <li>Export conversation history</li>
            <li>Advanced analytics and insights</li>
            <li>Custom embedding models selection</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box" style="text-align: center; margin-top: 2rem;">
        <h4 style="color: #667eea;">Built with LangChain, Streamlit, and HuggingFace</h4>
        <p style="color: #666;">Powered by advanced RAG techniques and state-of-the-art NLP models</p>
    </div>
    """, unsafe_allow_html=True)
# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    """Main application entry point"""
    # Initialize session state
    init_session_state()
    
    # Navigation in sidebar (KEEP IT SIMPLE)
    st.sidebar.markdown("## 🧭 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["💬 Chat", "🛠️ Tech Stack", "ℹ️ About"],
        label_visibility="collapsed"
    )

    # Track current page
    st.session_state.current_page = page
    
    # Route to appropriate page
    if page == "💬 Chat":
        show_chat_page()
    elif page == "🛠️ Tech Stack":
        show_tech_stack()
    elif page == "ℹ️ About":
        show_about_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        <p><strong>YouTube Video Chatbot</strong></p>
        <p>v1.0.0 | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# RUN APPLICATION
# ============================================================
if __name__ == "__main__":
    main()
