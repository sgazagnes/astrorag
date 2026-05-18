import gradio as gr
from src.retriever import load_retriever, build_qa_chain, query

# Load once at startup
retriever = load_retriever()
chain = build_qa_chain(retriever)


def answer(question: str, history: list) -> str:
    """Return answer string — ChatInterface handles history format."""
    if not question.strip():
        return ""
    result = query(chain, retriever, question)
    sources = "\n".join([
        f"• {s.split('/')[-1].split(chr(92))[-1]}"
        for s in result["sources"]
    ])
    return f"{result['answer']}\n\n**Sources:**\n{sources}"


demo = gr.ChatInterface(
    fn=answer,
    title="🔭 AstroRAG",
    description=(
        "A RAG chatbot over CLASSY survey research papers. "
        "Ask questions about UV spectroscopy, Lyman-alpha emission, "
        "star-forming galaxies, and ionizing photon escape."
    ),
    examples=[
        "What are the main Lyman-alpha escape mechanisms?",
        "What is the CLASSY survey and what does it observe?",
        "How is the UV spectral slope related to dust attenuation?",
    ]
)

if __name__ == "__main__":
    print(gr.__version__)
    demo.launch()