# IntelliDoc  
**An Agentic RAG System for Intelligent Document Question-Answering**

**IntelliDoc** is a two-phase, AI-powered pipeline that ingests structured and unstructured documents (PDF, DOCX, PPTX, TXT), semantically indexes them using embeddings, and enables conversational querying backed by reliable citationsn.

Built entirely from scratch (without LangChain), IntelliDoc implements key agentic behaviors like multi-step retrieval, reranking, adaptive tool use, and multimodal reasoning.

---

## Key Features

- **Multi-document Ingestion**  
  Extracts structured content (paragraphs, titles, tables, images) using layout-aware parsing.

- **Semantic Retrieval + Re-ranking**  
  Finds top-relevant information from embedded chunks using Cohere embeddings + rerank API.

- **Multimodal Understanding**  
  Visual content (charts, tables, figures) is processed via Gemini 1.5 Flash for accurate, searchable Q&A.

- **LLM-based Answering with Citations**  
  Gemini or Groq LLMs provide grounded answers, preserving `[document_name.pdf, page X]` citations.

- **Agentic Workflow (Custom Built)**  
  Each query follows a reasoning plan : query expansion → retrieve → rerank → prompt → generate.

- **Interactive QA Loop**  
  Ask multi-turn questions to your knowledge base, track stats, and get query-specific feedback prompts.

- **Secure Config via `.env`**  
  All keys are managed privately using environment variables.

---

## ⚙How It Works

### Phase 1 – Ingest Documents


This will:
- Parse documents from `data/documents/`
- Segment them into meaningful chunks
- Add chunk metadata (source, type, page number, etc.)
- Embed and store chunks in ChromaDB for fast retrieval

---

### Phase 2 – Ask Questions


This enables:
- Conversational Q&A over all documents stored
- Searchable answers enhanced with proper citations
- Adaptive query refinement and reranking for best context
- Feedback loop for rating system responses (`y / n / skip`)

---

## Technologies Used

| Component        | Technology                           |
|------------------|----------------------------------------|
| **LLMs**         | Gemini 1.5 Flash (Vision), Groq LLaMa 3 |
| **Embedding Model** | Cohere `embed-english-v3.0`         |
| **Vector Store** | ChromaDB                              |
| **Parser**       | unstructured.io (PDF, DOCX, etc.)      |
| **Reranker**     | Cohere `rerank-english-v3.0`          |

---

## Setup Instructions

### Clone the project

git clone https://github.com/your-username/intellidoc.git

cd intellidoc

### Create a virtual environment

python -m venv venv

source venv/bin/activate 

##### Windows : venv\Scripts\activate


### Install dependencies

pip install -r requirements.txt


### Configure your environment variables

Create a `.env` file in the root directory:

`COHERE_API_KEY=your_cohere_api_key`

`GEMINI_API_KEY=your_gemini_api_key`

`GROQ_API_KEY=your_groq_api_key`


Add your source documents (`.pdf`) into : `/data/documents/`


You're now ready to roll!

---

## Example Use Cases

- Summarize technical reports and research papers.
- Ask “What, Why, How” questions grounded in project documentation.
- Retrieve key takeaways with page-level citations.

---

## Developer

### **Aarav Raj**  
B.Tech Computer Science and Engineering  
PES University, Bangalore  
GitHub: [@Raj-Aarav](https://github.com/Raj-Aarav)

---

