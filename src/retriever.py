from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

INDEX_DIR = Path("faiss_index")

PROMPT_TEMPLATE = """You are an astrophysics research assistant \
with expertise in UV spectroscopy, star-forming galaxies, and the \
CLASSY survey. Use the following excerpts from research papers to \
answer the question accurately and concisely. If the answer is not \
in the provided context, say so clearly rather than speculating.

Context:
{context}

Question: {question}

Answer:"""


def load_retriever(index_dir: Path = INDEX_DIR):
    """Load FAISS index and return a retriever."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(retriever):
    """Build the RAG chain using LCEL."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def query(chain, retriever, question: str) -> dict:
    """Run a query and return answer + sources."""
    # Get source documents separately
    source_docs = retriever.invoke(question)
    answer = chain.invoke(question)
    sources = list({
        doc.metadata.get("source", "unknown")
        for doc in source_docs
    })
    return {
        "answer": answer,
        "sources": sources
    }