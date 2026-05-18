# AstroRAG

A retrieval-augmented generation (RAG) system for querying astrophysics research papers. Built over the CLASSY survey corpus, it allows researchers to ask natural language questions and receive answers grounded in the literature, with source attribution.

## Overview

AstroRAG ingests PDF research papers, chunks and embeds them using OpenAI embeddings, stores the index locally with FAISS, and retrieves relevant context at query time to generate accurate, citation-backed answers via GPT-4o-mini.

Three interfaces are provided: a browser-based chat UI, a REST API, and a terminal client.

## Setup

**Requirements:** Python 3.11, conda

```bash
conda create -n astrorag python=3.11
conda activate astrorag
pip install -r requirements.txt
```

Create a `.env` file at the project root:
```
OPENAI_API_KEY=your_key_here
```

## Usage

### 1. Ingest papers

Place PDF files in `data/papers/`, then build the FAISS index:
```bash
python -m src.ingest
```
This only needs to be run once, or when new papers are added.

### 2. Launch the chat UI

```bash
python app.py
```
Opens a Gradio interface at `http://localhost:7860`.

### 3. Launch the API

```bash
python api.py
```
Starts a FastAPI server at `http://localhost:8000`.

Query from terminal:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the CLASSY survey?"}'
```

Interactive documentation available at `http://localhost:8000/docs`.

### 4. Terminal client

```bash
# Single question
python client.py "What are the main Lyman-alpha escape mechanisms?"

# Interactive session
python client.py
```

### 5. Evaluation

```bash
python -m src.evaluate
```
Runs 10 domain-specific questions against the pipeline and scores retrieval precision and answer quality. Results saved to `eval_results.json`.

## Project Structure

```
astrorag/
├── data/
│   └── papers/          # PDF papers (not committed)
├── src/
│   ├── ingest.py        # Load, chunk, embed, save index
│   ├── retriever.py     # Load index, build RAG chain
│   └── evaluate.py      # Evaluation suite
├── faiss_index/         # Generated index (not committed)
├── app.py               # Gradio chat interface
├── api.py               # FastAPI REST endpoint
├── client.py            # Terminal client
├── requirements.txt
└── .env                 # API keys (not committed)
```

## Corpus

The current index covers 15 papers from the CLASSY (COS Legacy Archive Spectroscopic SurveY) collaboration, focusing on UV spectroscopy of local star-forming galaxies as analogues of high-redshift reionization-era objects.

## Technical notes

- Chunk size: 600 tokens with 80-token overlap
- Embedding model: `text-embedding-3-small`
- LLM: `gpt-4o-mini` at temperature 0
- Retrieval: top-5 chunks by cosine similarity
- Evaluation: LLM-as-judge scoring (0-3) + keyword-based retrieval precision