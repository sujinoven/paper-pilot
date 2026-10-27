# 📚 PaperPilot — Research Paper Q&A Assistant

**PaperPilot** is a local, AI-powered research paper question-answering application built using Retrieval-Augmented Generation (RAG). It allows users to upload PDF research papers, index their contents, and ask natural-language questions with answers grounded in retrieved passages.

The project combines PDF parsing, semantic search, vector databases, local large language models, and a Streamlit interface.

## ✨ Features

- **PDF Upload:** Upload and index research papers through a web interface.
- **PDF Text Extraction:** Extract text, paper titles, section headings, and page information using PyMuPDF.
- **Section-Aware Chunking:** Divide research papers into overlapping chunks while preserving document metadata.
- **Semantic Search:** Retrieve relevant passages using sentence-transformer embeddings.
- **Hybrid Re-ranking:** Combine semantic similarity, keyword overlap, and section-heading relevance.
- **Local LLM Integration:** Generate answers using Qwen2.5 through Ollama.
- **Source Citations:** Display numbered citations and expandable source passages with page and section details.
- **Citation Validation:** Check whether generated citation numbers correspond to retrieved passages.
- **Grounded Answer Generation:** Encourage evidence-based answers and refusals when information is insufficient.
- **Streamlit Interface:** Interact with the RAG pipeline through a browser-based chat application.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.11 |
| Frontend | Streamlit |
| PDF processing | PyMuPDF |
| Embedding model | BAAI/bge-small-en-v1.5 |
| Vector database | ChromaDB |
| Language model | Qwen2.5 3B |
| Local LLM runtime | Ollama |
| Package management | uv |
| Testing | pytest |

## 🏗️ System Architecture

```text
                  Research Paper (PDF)
                          |
                          v
                  PDF Text Extraction
                       PyMuPDF
                          |
                          v
                  Section-Aware Chunking
                          |
                          v
                   Text Embeddings
                BGE-small-en-v1.5
                          |
                          v
                       ChromaDB
                          |
                          |
User Question ------------+
       |
       v
Question Embedding
       |
       v
Semantic Retrieval
       |
       v
Hybrid Re-ranking
       |
       v
Relevant Paper Passages
       |
       v
Context Construction
       |
       v
Qwen2.5 3B (Ollama)
       |
       v
Citation Validation
       |
       v
Answer with Source Citations
       |
       v
Streamlit Chat Interface
```

## 📂 Project Structure

```text
paperpilot-project/
│
├── app.py
├── pyproject.toml
├── uv.lock
│
├── data/
│   ├── papers/
│   └── chroma/
│
├── src/
│   └── paperpilot/
│       ├── ingest.py
│       ├── chunking.py
│       ├── vector_store.py
│       ├── retrieval.py
│       ├── context.py
│       ├── citations.py
│       ├── generation.py
│       └── evaluation.py
│
├── tests/
│
├── test_rag_manual.py
├── test_evaluation_manual.py
├── evaluate_rag_manual.py
│
└── README.md
```

## ⚙️ Installation and Setup

### 1. Prerequisites

Install the following:

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)

### 2. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/paperpilot-project.git
cd paperpilot-project
```

Replace `YOUR_USERNAME` with your GitHub username.

### 3. Install Dependencies

```bash
uv sync
```

### 4. Download the Local Language Model

```bash
ollama pull qwen2.5:3b
```

Ensure Ollama is running and accessible at `http://localhost:11434`.

### 5. Launch PaperPilot

```bash
uv run streamlit run app.py
```

Open the URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## 🚀 How to Use

1. Launch the Streamlit application.
2. Upload a text-based research paper in PDF format.
3. Click **Index Paper** to extract, chunk, embed, and store its content.
4. Enter a question in the chat interface.
5. Review the generated answer and expand the source passages to inspect the supporting evidence.

**Example questions:**

- Why does the Transformer use multi-head attention?
- What positional encodings are used in the Transformer?
- What optimizer was used during training?
- What is masked language modeling?

## 🔍 Retrieval and Answer Generation

PaperPilot uses the embedding model `BAAI/bge-small-en-v1.5` to convert paper chunks and user questions into vector representations.

The initial retrieval stage searches ChromaDB using cosine similarity.

A custom re-ranking function combines three signals:

```text
Combined Score =
    0.65 × Semantic Similarity
  + 0.15 × Keyword Overlap
  + 0.20 × Section Relevance
```

The highest-ranked passages are formatted into a numbered context and provided to the local Qwen2.5 model.

The generation pipeline instructs the model to answer using retrieved evidence, cite relevant passages, and refuse questions that cannot be answered from the supplied context.

Citation checks validate citation identifiers and check whether a non-refusal answer contains at least one valid citation. They do not independently prove that every claim is supported.

## 🧪 Testing and Evaluation

Run the automated tests:

```bash
uv run pytest -q
```

Run the retrieval evaluation:

```bash
uv run python test_evaluation_manual.py
```

Run the end-to-end RAG evaluation:

```bash
uv run python evaluate_rag_manual.py
```

### Evaluation Results

The latest recorded evaluation used 12 questions against the research paper *Attention Is All You Need*, including nine answerable questions and three unsupported questions.

| Metric | Recorded Result |
|---|---|
| Automated tests | 69 passed |
| Retrieval Hit Rate@3 | 9/9 (100%) |
| Citation Coverage Pass Rate | 9/9 (100%) |
| Refusal Correctness | 12/12 (100%) |
| Average Keyword Coverage | 94.4% |
| Answers flagged by configured quality checks | 0/12 |

These results reflect a small, paper-specific evaluation set and should not be interpreted as general accuracy guarantees. Some answers can still be incomplete, circular, or insufficiently grounded even when automated checks pass.

## ⚠️ Current Limitations

- Retrieval currently searches the shared vector collection, so evidence from multiple indexed papers may be mixed.
- Strict paper-specific retrieval filtering has not yet been implemented.
- Scanned or image-only PDFs may require OCR.
- Mathematical expressions and tables may not always be extracted correctly.
- Generated answers may occasionally contain unsupported or incomplete explanations.
- Citation validation checks reference identifiers rather than independently verifying factual support.
- Uploaded PDF filenames may collide if different files share the same name.
- The application requires a locally running Ollama instance.

## 🔮 Future Improvements

- Paper-specific retrieval and document selection
- Multi-paper comparison and summarization
- Improved mathematical equation and table extraction
- Stronger answer-grounding verification
- Automated RAG evaluation across multiple research papers
- Docker deployment
- Improved upload management and document metadata

## 🎯 Project Objective

PaperPilot was developed as a hands-on project to explore the implementation of a complete RAG pipeline, including document ingestion, vector embeddings, semantic retrieval, prompt engineering, citation handling, local LLM inference, and evaluation.

The goal is to make research papers easier to explore while providing traceable evidence for generated answers.

## 📄 License

No license has been specified yet. Add a `LICENSE` file if you want to define how others may use, modify, or distribute this project.
