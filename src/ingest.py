import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_DIR = Path("data/papers")
INDEX_DIR = Path("faiss_index")


def load_documents(data_dir: Path) -> list:
    """Load all PDFs from the data directory."""
    docs = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        print(f"Loading: {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    print(f"Loaded {len(docs)} pages from {data_dir}")
    return docs


def chunk_documents(docs: list) -> list:
    """
    Split documents into chunks.
    chunk_size=600 works well for scientific papers —
    large enough to preserve context, small enough
    for precise retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    return chunks


def build_index(chunks: list) -> FAISS:
    """Embed chunks and build FAISS index."""
    print("Building embeddings — this will take a minute...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # cheap and accurate
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_index(vectorstore: FAISS, index_dir: Path):
    """Save FAISS index to disk."""
    index_dir.mkdir(exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print(f"Index saved to {index_dir}")


if __name__ == "__main__":
    docs = load_documents(DATA_DIR)
    chunks = chunk_documents(docs)
    vectorstore = build_index(chunks)
    save_index(vectorstore, INDEX_DIR)
    print("Ingestion complete.")