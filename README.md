# 🎥 Advanced RAG-Based YouTube Video Chatbot

An intelligent conversational AI system that enables natural language interactions with YouTube video content using advanced Retrieval-Augmented Generation (RAG) techniques, hybrid search, and multi-query retrieval.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-0.1+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Table of Contents
- [Live Demo](#demo)
- [Overview](#overview)
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

### Live Demo

[*Link to deployed application if available*](https://youtube-chatbot-9v8bdeuhfkfcet7njtsxa4.streamlit.app/)

## 🎯 Overview

The YouTube Video Chatbot is a sophisticated AI-powered application that transforms how users interact with video content. Instead of watching entire videos to find specific information, users can ask questions in natural language and receive accurate, context-aware answers extracted directly from video transcripts. The system employs state-of-the-art natural language processing techniques including hybrid retrieval, multi-query generation, and cross-encoder reranking.

### Problem Statement

- **Time-Consuming**: Watching lengthy videos to find specific information
- **Inefficient Navigation**: Difficulty in locating relevant segments
- **Limited Accessibility**: Inability to quickly reference video content

### Solution

An intelligent chatbot that:
- Processes YouTube video transcripts automatically
- Answers questions with pinpoint accuracy
- Maintains conversational context
- Provides an embedded video player for reference

## ✨ Key Features

### 🔍 Advanced Retrieval System

- **Hybrid Search**: Combines semantic (FAISS) and keyword (BM25) search for optimal retrieval
- **Multi-Query Generation**: Automatically generates multiple query variations to improve coverage
- **Cross-Encoder Reranking**: Re-ranks retrieved documents using transformer-based models for relevance
- **Configurable Alpha Parameter**: Adjustable weighting between semantic and keyword search (default: 0.5)

### 💬 Intelligent Conversation

- **Context-Aware Responses**: Maintains conversation history for natural follow-up questions
- **Memory Management**: Automatically manages conversation history (last 5 exchanges)
- **Natural Language Understanding**: Powered by Mistral-7B-Instruct model
- **Detailed Source Citations**: Shows generated query variations for transparency

### 🎨 User Experience

- **Modern Gradient UI**: Beautiful purple-blue gradient design with glassmorphism effects
- **Embedded Video Player**: Watch videos while chatting for better context
- **Responsive Layout**: Two-column design with video and chat side-by-side
- **Real-Time Processing**: Instant answers with loading indicators
- **Navigation System**: Multiple pages (Chat, Tech Stack, About)

### ⚡ Performance Optimization

- **Caching**: Uses `@st.cache_resource` for model and index caching
- **Efficient Indexing**: FAISS for fast vector similarity search
- **Optimized Chunking**: Smart text splitting with overlap for context preservation
- **Resource Management**: Memory-efficient processing pipeline

## 🏗️ Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Streamlit)              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Video Input │  │Chat Interface│  │ Tech Details │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                     RAG Pipeline                            │
│                                                             │
│  ┌────────────────────────────────────────────────┐         │
│  │         Multi-Query Generation                 │         │
│  │  (Generates 3 query variations using LLM)      │         │
│  └────────────────────────────────────────────────┘         │
│                          ↓                                  │
│  ┌────────────────────────────────────────────────┐         │
│  │          Hybrid Retrieval System               │         │
│  │  ┌──────────────┐    ┌──────────────┐          │         │
│  │  │ FAISS Search │    │ BM25 Search  │          │         │
│  │  │  (Semantic)  │    │  (Keyword)   │          │         │
│  │  └──────────────┘    └──────────────┘          │         │
│  │           ↓                  ↓                 │         │
│  │          Score Fusion (Alpha = 0.5)            │         │
│  └────────────────────────────────────────────────┘         │
│                          ↓                                  │
│  ┌────────────────────────────────────────────────┐         │
│  │        Cross-Encoder Reranking                 │         │
│  │  (ms-marco-MiniLM-L-6-v2)                      │         │
│  └────────────────────────────────────────────────┘         │
│                          ↓                                  │
│  ┌────────────────────────────────────────────────┐         │
│  │         Response Generation                    │         │
│  │  (Mistral-7B-Instruct with context)            │         │
│  └────────────────────────────────────────────────┘         │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                   Data Processing Layer                     │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │Transcript│→ │ Chunking │→ │Embeddings│→ │  FAISS   │     │
│  │Extraction│  │          │  │Generation│  │  Index   │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
│                                     ↓                       │
│                              ┌──────────┐                   │
│                              │   BM25   │                   │
│                              │  Index   │                   │
│                              └──────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### Retrieval Process Flow

1. **Query Expansion**: User question → Multi-query generator → 3 query variations
2. **Parallel Retrieval**: Each query runs through:
   - Semantic search (FAISS with embeddings)
   - Keyword search (BM25 algorithm)
3. **Score Fusion**: Combines scores with configurable alpha
4. **Deduplication**: Removes duplicate retrieved chunks
5. **Reranking**: Cross-encoder scores all candidates
6. **Context Assembly**: Top-K documents formatted for LLM
7. **Generation**: LLM generates response with conversation history

## 🛠️ Technologies Used

### Core Frameworks

- **[Streamlit](https://streamlit.io/)** (1.28+): Web application framework
- **[LangChain](https://python.langchain.com/)** (0.1+): LLM application framework
- **Python** (3.8+): Programming language

### NLP & Machine Learning

- **[HuggingFace Transformers](https://huggingface.co/)**: Model inference and embeddings
  - **LLM**: `mistralai/Mistral-7B-Instruct-v0.2` (7B parameters)
  - **Embeddings**: `BAAI/bge-base-en-v1.5` (768 dimensions)
  - **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **[Sentence Transformers](https://www.sbert.net/)**: Cross-encoder reranking
- **[FAISS](https://github.com/facebookresearch/faiss)**: Facebook AI Similarity Search
- **[Rank-BM25](https://github.com/dorianbrown/rank_bm25)**: BM25 algorithm implementation

### Data Processing

- **[youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)**: Transcript extraction
- **NumPy**: Numerical computing
- **LangChain Text Splitters**: Intelligent document chunking

### Development Tools

- **Git**: Version control
- **pip**: Package management
- **Virtual Environment**: Dependency isolation

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- HuggingFace API token (free tier available)
- Git

### Step-by-Step Setup

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/youtube-video-chatbot.git
cd youtube-video-chatbot
```

2. **Create Virtual Environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Set Up HuggingFace Token**

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
```

Or set as environment variable:

```bash
# Windows
set HF_TOKEN=your_token_here

# macOS/Linux
export HF_TOKEN=your_token_here
```

5. **Run the Application**

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Requirements.txt

```txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-community>=0.0.10
langchain-huggingface>=0.0.1
youtube-transcript-api>=0.6.1
faiss-cpu>=1.7.4
rank-bm25>=0.2.2
sentence-transformers>=2.2.2
numpy>=1.24.0
python-dotenv>=1.0.0
```

## 🚀 Usage

### Quick Start

1. **Launch the Application**
   ```bash
   streamlit run app.py
   ```

2. **Enter Video URL**
   - Navigate to the sidebar
   - Paste a YouTube video URL or ID
   - Click "🚀 Initialize Chatbot"

3. **Wait for Processing**
   - System fetches transcript
   - Creates embeddings and indexes
   - Typically takes 10-30 seconds

4. **Start Chatting**
   - Type your question in the chat input
   - Receive AI-generated answers
   - Ask follow-up questions naturally

### Example Queries

**General Understanding:**
- "What is this video about?"
- "Summarize the main points discussed"

**Specific Information:**
- "What does the speaker say about [topic]?"
- "Can you explain the concept of [term] mentioned in the video?"

**Detailed Analysis:**
- "What are the steps to [process]?"
- "Compare [concept A] and [concept B] as discussed"

**Follow-up Questions:**
- "Can you elaborate on that?"
- "What examples were given?"

### Pre-loaded Example Videos

The application includes three educational videos:

1. **Attention Mechanism** (`kCc8FmEb1nY`)
2. **Neural Networks** (`aircAruvnKk`)
3. **Transformers** (`zxQyTK8quyY`)

Click "Load Video" buttons on the welcome page to try them instantly.

## ⚙️ Configuration

### System Parameters

Located in `Config` class in `app.py`:

```python
class Config:
    # Chunking Parameters
    CHUNK_SIZE = 1000          # Characters per chunk
    CHUNK_OVERLAP = 200        # Overlap between chunks
    
    # Retrieval Parameters
    RETRIEVAL_K = 10           # Initial retrieval count
    FINAL_K = 4                # Final documents after reranking
    ALPHA = 0.5                # Semantic vs keyword weight (0-1)
    
    # Model Configuration
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Conversation Settings
    MAX_HISTORY = 5            # Number of Q&A pairs to remember
```

### Customization Options

**Adjust Retrieval Quality:**
- Increase `RETRIEVAL_K` for more comprehensive search
- Increase `FINAL_K` for more context in responses
- Tune `ALPHA` (higher = more semantic, lower = more keyword-based)

**Modify Chunking:**
- Larger `CHUNK_SIZE` for broader context
- Increase `CHUNK_OVERLAP` for better context preservation

**Change Models:**
- Replace model names with any compatible HuggingFace models
- Ensure models support the required interfaces

### LLM Parameters

In `setup_chatbot()` function:

```python
llm = HuggingFaceEndpoint(
    repo_id=Config.LLM_MODEL,
    max_new_tokens=512,      # Maximum response length
    temperature=0.7,         # Creativity (0.0-1.0)
    top_p=0.95              # Nucleus sampling
)
```

## 📁 Project Structure

```
youtube-video-chatbot/
│
├── app.py                      # Main application file
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .env                        # Environment variables (not in repo)
├── .gitignore                 # Git ignore rules
│
├── assets/                    # Static assets (optional)
│   ├── screenshots/
│   └── diagrams/
│
└── docs/                      # Additional documentation
    ├── ARCHITECTURE.md
    ├── API.md
    └── CONTRIBUTING.md
```

### Key Components in app.py

```
app.py (1,200+ lines)
│
├── PAGE CONFIG & STYLING
│   ├── Streamlit configuration
│   └── Custom CSS styling
│
├── CONFIGURATION
│   └── Config class with all parameters
│
├── UTILITY FUNCTIONS
│   ├── fetch_youtube_transcript()
│   ├── chunk_transcript()
│   └── extract_video_id()
│
├── CORE CLASSES
│   ├── HybridRetriever
│   │   ├── __init__()
│   │   └── retrieve()
│   │
│   └── MultiQueryRetriever
│       ├── __init__()
│       ├── generate_queries()
│       └── retrieve()
│
├── RAG PIPELINE
│   ├── format_docs()
│   └── create_rag_chain()
│
├── SESSION MANAGEMENT
│   ├── init_session_state()
│   └── setup_chatbot() [Cached]
│
├── PAGE COMPONENTS
│   ├── show_chat_page()
│   ├── show_tech_stack()
│   └── show_about_page()
│
└── MAIN APPLICATION
    └── main()
```

## 🔬 How It Works

### 1. Transcript Extraction

```python
# Fetch transcript from YouTube
transcript_api = YouTubeTranscriptApi()
transcript = transcript_api.fetch(video_id, languages=['en'])
text = " ".join([entry["text"] for entry in transcript])
```

### 2. Text Chunking

```python
# Split into manageable chunks with overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.create_documents([transcript])
```

### 3. Embedding Generation

```python
# Create vector embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={'normalize_embeddings': True}
)
vector_store = FAISS.from_documents(chunks, embeddings)
```

### 4. Hybrid Retrieval

```python
# Semantic search
semantic_results = vector_store.similarity_search(query, k=10)

# Keyword search
bm25_scores = bm25_index.get_scores(tokenized_query)

# Fusion
combined_score = alpha * semantic_score + (1-alpha) * bm25_score
```

### 5. Reranking

```python
# Cross-encoder reranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs = [[query, doc.content] for doc in candidates]
scores = reranker.predict(pairs)
top_docs = sorted_by_score(docs, scores)[:4]
```

### 6. Response Generation

```python
# Generate answer with context
prompt = f"""
Context: {formatted_docs}
History: {conversation_history}
Question: {user_question}
"""
response = llm.invoke(prompt)
```


## 🚀 Future Enhancements

### Planned Features

- [ ] **Multi-language Support**: Process non-English transcripts
- [ ] **Timestamp References**: Link answers to specific video timestamps
- [ ] **Playlist Support**: Analyze multiple videos simultaneously
- [ ] **Export Functionality**: Download conversation history
- [ ] **Advanced Analytics**: Video content insights and statistics
- [ ] **Custom Models**: User-selectable embedding and LLM models
- [ ] **Video Comparison**: Compare content across multiple videos
- [ ] **Bookmark System**: Save important Q&A exchanges
- [ ] **API Endpoint**: RESTful API for programmatic access
- [ ] **Mobile Optimization**: Enhanced mobile UI/UX

### Technical Improvements

- [ ] **Async Processing**: Parallel query processing
- [ ] **Database Integration**: Persistent storage for processed videos
- [ ] **User Authentication**: Personal conversation history
- [ ] **Rate Limiting**: API usage management
- [ ] **Performance Monitoring**: Analytics dashboard
- [ ] **A/B Testing**: Model comparison framework

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Add docstrings to functions and classes
- Include comments for complex logic

### Testing

- Test new features thoroughly
- Ensure existing functionality still works
- Document test cases in PR description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Contact

**Praanshull Verma**


## 🙏 Acknowledgments

- **LangChain** for the excellent RAG framework
- **HuggingFace** for providing open-source models
- **Streamlit** for the intuitive web framework
- **YouTube Transcript API** for transcript extraction
- **Facebook AI Research** for FAISS
- **Sentence Transformers** for the reranking models

## 📊 Project Statistics

- **Lines of Code**: ~1,200
- **Components**: 6 major classes
- **Functions**: 15+ utility functions
- **API Integrations**: 3 (YouTube, HuggingFace, Streamlit)
- **Models Used**: 3 (LLM, Embeddings, Reranker)

---

⭐ **Star this repository** if you find it helpful!

🐛 **Report bugs** by opening an issue

💡 **Suggest features** through discussions

📖 **Read the docs** in the `/docs` folder

Built using Python, LangChain, and Streamlit
