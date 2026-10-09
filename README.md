<<<<<<< HEAD
# ReAct-Based RAG Agent with Google Gemini

An autonomous, agentic **Retrieval-Augmented Generation (RAG)** system powered by the **ReAct (Reasoning and Action)** framework and Google Gemini (`gemini-3.8-flash` and `gemini-embedding-001`).

---

## 🌟 Key Features

- **Agentic ReAct Architecture**: Instead of basic single-shot RAG, the agent uses a transparent **Thought ➔ Action ➔ Observation ➔ Reflection** loop to decide what information to search, inspect pages, and verify evidence.
- **RAG Tools Suite**:
  - `search_document(query, top_k)`: Dense semantic embeddings + lexical keyword matching with cosine similarity ranking.
  - `read_page(page_number)`: Retrieves the complete text of any individual page for comprehensive context.
  - `get_document_info()`: Analyzes document metadata, total pages, character counts, and section previews.
- **Auditable Citations**: Strict protocol ensuring every factual statement cites exact page numbers (e.g. `[Page 2]`).
- **Disk Caching**: SHA-256 hashed vector embeddings cache in `.rag_cache/` so embeddings are computed only once unless the PDF changes.
- **Developer-Friendly CLI**: Color-coded live reasoning traces showing the agent's internal thought process.

---

## 📁 Project Structure

```
rag_react_agent/
├── .env                  # Your Gemini API Key configuration
├── .env.example          # Template configuration
├── requirements.txt      # Python dependencies
├── config.py             # Settings, paths, and environment validation
├── pdf_loader.py         # PDF parsing and overlapping chunking engine
├── vector_store.py       # Local vector store with cosine search & caching
├── rag_tools.py          # Tools exposed to the ReAct agent
├── react_agent.py        # ReAct reasoning loop & tool orchestration
├── create_sample_pdf.py  # Utility to generate test multi-page PDF
├── main.py               # Interactive CLI and query runner
└── rag_file.pdf          # Your target PDF document (placed in root)
```

---

## 🚀 Quickstart Guide

### 1. Set Workspace & Virtual Environment

Open a terminal in the project directory:
```powershell
cd C:\Users\nprhp\.gemini\antigravity\scratch\rag_react_agent
```

Activate the virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

If you ever need to reinstall dependencies:
```powershell
.\.venv\Scripts\pip install -r requirements.txt
```

---

### 2. Configure Your Gemini API Key

1. Open the `.env` file in the project folder.
2. Replace `YOUR_GEMINI_API_KEY_HERE` with your actual Gemini API key from [Google AI Studio](https://aistudio.google.com/):

```env
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-3.8-flash
EMBEDDING_MODEL=gemini-embedding-001
PDF_PATH=rag_file.pdf
```

---

### 3. Place Your PDF Document

Place your document in the project root folder named **`rag_file.pdf`**:
```
C:\Users\nprhp\.gemini\antigravity\scratch\rag_react_agent\rag_file.pdf
```

*(Note: If `rag_file.pdf` is not found, the system will automatically generate a sample 3-page research paper PDF so you can test the pipeline immediately!)*

---

### 4. Run the Agent

#### Run System Diagnostic Check:
```powershell
.\.venv\Scripts\python main.py --check
```

#### Run an Interactive Session:
```powershell
.\.venv\Scripts\python main.py
```
Then ask any question, for example:
- *"What is the main conclusion of the document?"*
- *"What were the benchmark results and accuracy metrics?"*
- *"Summarize the methodology on page 2."*

#### Run a Single Question:
```powershell
.\.venv\Scripts\python main.py --query "What is the hallucination rate reported in the experiments?"
```

#### Force Re-index of Embeddings:
```powershell
.\.venv\Scripts\python main.py --reindex
```

---

## 🧠 The ReAct Reasoning Flow

When you ask a question, the agent follows this sequence:

```
User Query: "What was the accuracy improvement of Titan?"
  │
  ▼
🧠 Thought: "I need to check the benchmark numbers in the results section.
             I will search for 'retrieval precision and accuracy'."
  │
  ▼
⚡ Action: search_document(query="retrieval precision and accuracy", top_k=3)
  │
  ▼
👁️ Observation: "[Passage 1 | Page 3 | Relevance: 0.88]
                 Retrieval Precision@3: 94.8% (up from 78.2% in naive single-shot RAG)..."
  │
  ▼
🧠 Thought: "I have the exact numbers from Page 3. I will now synthesize the final answer."
  │
  ▼
🎯 Final Answer: "Based on [Page 3], Project Titan achieved a Retrieval Precision@3 of 94.8%,
                 compared to 78.2% in naive single-shot RAG..."
```
=======
# rag-react-agent
A Python-based RAG (Retrieval-Augmented Generation) and ReAct AI agent that processes PDF documents, retrieves relevant information using a vector store, and generates context-aware answers using an LLM.
>>>>>>> aacd39403771d3c1bee4774881212c815c989995
