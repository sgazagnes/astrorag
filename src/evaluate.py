"""
AstroRAG Evaluation Script
Usage: python -m src.evaluate

Evaluates the RAG pipeline on 10 domain-specific questions using:
1. Retrieval precision  — did the right sources come back?
2. Answer relevance     — does the LLM answer actually address the question?
                          scored by a separate LLM judge call (0-3 scale)

Results saved to eval_results.json
"""

import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.retriever import load_retriever, build_qa_chain, query

load_dotenv()

# ── EVALUATION SET ────────────────────────────────────────────────────────────
# Fill in expected_answer with a short 1-2 sentence ground truth.
# relevant_keywords: terms that SHOULD appear in retrieved chunks.

EVAL_SET = [
    {
        "id": 1,
        "question": "What is the CLASSY survey and what are its main scientific goals?",
        "expected_answer": "CLASSY is a Hubble Space Telescope survey of 45 local star-forming galaxies designed to study UV spectral properties of low-redshift analogues of high-redshift galaxies that may have reionized the universe.",
        "relevant_keywords": ["CLASSY", "reionization", "UV", "star-forming"]
    },
    {
        "id": 2,
        "question": "How many galaxies are in the CLASSY sample and what are the selection criteria?",
        "expected_answer": "The CLASSY sample contains 45 star-forming galaxies selected to span a wide range of physical properties including star formation rate, metallicity, and UV luminosity to represent diverse conditions in the local universe.",
        "relevant_keywords": ["45", "sample", "selection", "star formation", "metallicity"]
    },
    {
        "id": 3,
        "question": "What are the main physical properties of CLASSY galaxies such as star formation rate, metallicity, and stellar mass?",
        "expected_answer": "CLASSY galaxies span a wide range of star formation rates, stellar masses, and metallicities, with many being low-metallicity, high-ionization compact galaxies similar to early universe objects.",
        "relevant_keywords": ["star formation rate", "metallicity", "stellar mass", "ionization"]
    },
    {
        "id": 4,
        "question": "What role does the neutral hydrogen column density play in Lyman-alpha escape?",
        "expected_answer": "Higher neutral hydrogen column density increases the number of resonant scatterings of Lyman-alpha photons, reducing the escape fraction. Lower HI column density, often associated with outflows or ionized channels, facilitates escape.",
        "relevant_keywords": ["neutral hydrogen", "column density", "Lyman-alpha", "escape", "HI"]
    },
    {
        "id": 5,
        "question": "How does the ionization parameter affect UV emission line diagnostics?",
        "expected_answer": "Higher ionization parameters produce stronger high-ionization UV emission lines such as C IV and He II, making them sensitive diagnostics of the hardness of the ionizing radiation field and the physical conditions of the gas.",
        "relevant_keywords": ["ionization parameter", "UV", "emission line", "C IV", "diagnostic"]
    },
    {
        "id": 6,
        "question": "What is the relationship between Lyman-alpha and Lyman continuum escape?",
        "expected_answer": "Lyman-alpha escape fraction is often used as an indirect tracer of Lyman continuum escape, as both are facilitated by low neutral hydrogen column density and high ionization. However the correlation is not perfect due to dust and geometry effects.",
        "relevant_keywords": ["Lyman-alpha", "Lyman continuum", "escape fraction", "tracer", "ionizing photons"]
    },
    {
        "id": 7,
        "question": "What are the main mechanisms driving outflows in CLASSY star-forming galaxies?",
        "expected_answer": "Outflows in CLASSY galaxies are primarily driven by stellar feedback including winds from massive stars and supernovae, which create channels of low neutral gas density that facilitate the escape of ionizing and Lyman-alpha photons.",
        "relevant_keywords": ["outflow", "feedback", "supernovae", "stellar winds", "neutral gas"]
    },
    {
        "id": 8,
        "question": "What UV absorption lines are used to trace the interstellar medium in CLASSY galaxies?",
        "expected_answer": "UV absorption lines such as Si II, C II, O I, and Al II are used to trace the neutral and low-ionization interstellar medium, while Si IV and C IV trace the higher-ionization phases and outflowing gas.",
        "relevant_keywords": ["Si II", "C II", "absorption", "interstellar medium", "UV"]
    },
    {
        "id": 9,
        "question": "How does dust attenuation affect the UV continuum slope in star-forming galaxies?",
        "expected_answer": "Dust attenuation reddens the UV continuum, steepening the spectral slope beta. Higher dust content produces redder (less negative) beta values, while young, low-dust galaxies show steep blue UV slopes.",
        "relevant_keywords": ["dust", "attenuation", "UV slope", "beta", "continuum"]
    },
    {
        "id": 10,
        "question": "What makes local CLASSY galaxies good analogues for high-redshift reionization-era galaxies?",
        "expected_answer": "CLASSY galaxies share key properties with high-redshift galaxies including compact morphology, low metallicity, high ionization, and strong UV emission lines, making them ideal local laboratories to study the physical processes that drove cosmic reionization.",
        "relevant_keywords": ["analogue", "high redshift", "reionization", "compact", "low metallicity"]
    },
]

# ── JUDGE PROMPT ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating the quality of an answer produced by a 
retrieval-augmented generation system over astrophysics research papers.

Question: {question}
Expected answer (ground truth): {expected_answer}
System answer: {system_answer}

Rate the system answer on a scale of 0 to 3:
0 - Completely wrong or irrelevant
1 - Partially correct but missing key information
2 - Mostly correct with minor gaps or inaccuracies  
3 - Correct and complete, matches the expected answer well

Reply with ONLY a JSON object in this exact format:
{{"score": <0-3>, "reason": "<one sentence explanation>"}}"""


# ── SCORING FUNCTIONS ─────────────────────────────────────────────────────────

def score_retrieval(source_docs: list, keywords: list) -> dict:
    """
    Check whether retrieved chunks contain the expected keywords.
    Returns hit rate: fraction of keywords found in retrieved context.
    """
    full_text = " ".join(
        doc.page_content.lower() for doc in source_docs
    )
    hits = [kw for kw in keywords if kw.lower() in full_text]
    return {
        "hits": hits,
        "misses": [kw for kw in keywords if kw.lower() not in full_text],
        "precision": round(len(hits) / len(keywords), 2)
    }


def score_answer(judge_llm, question: str, expected: str, answer: str) -> dict:
    """Use a separate LLM call to judge answer relevance (0-3 scale)."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        expected_answer=expected,
        system_answer=answer
    )
    response = judge_llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"score": -1, "reason": "Could not parse judge response."}


# ── MAIN EVALUATION LOOP ──────────────────────────────────────────────────────

def run_evaluation():
    print("\n AstroRAG Evaluation")
    print("=" * 60)

    retriever = load_retriever()
    chain = build_qa_chain(retriever)
    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    results = []
    total_retrieval = 0.0
    total_answer = 0.0
    valid_answer_scores = 0

    for item in EVAL_SET:
        print(f"\n[{item['id']}/10] {item['question'][:70]}...")

        # Get RAG response
        source_docs = retriever.invoke(item["question"])
        result = query(chain, retriever, item["question"])

        # Score retrieval
        ret_score = score_retrieval(source_docs, item["relevant_keywords"])

        # Score answer
        ans_score = score_answer(
            judge_llm,
            item["question"],
            item["expected_answer"],
            result["answer"]
        )

        total_retrieval += ret_score["precision"]
        if ans_score["score"] >= 0:
            total_answer += ans_score["score"]
            valid_answer_scores += 1

        print(f"  Retrieval precision : {ret_score['precision']:.0%} "
              f"({len(ret_score['hits'])}/{len(item['relevant_keywords'])} keywords)")
        print(f"  Answer score        : {ans_score['score']}/3 — {ans_score['reason']}")

        results.append({
            "id": item["id"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "system_answer": result["answer"],
            "sources": result["sources"],
            "retrieval": ret_score,
            "answer_score": ans_score
        })

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    avg_retrieval = total_retrieval / len(EVAL_SET)
    avg_answer = total_answer / valid_answer_scores if valid_answer_scores > 0 else 0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Avg retrieval precision : {avg_retrieval:.0%}")
    print(f"  Avg answer score        : {avg_answer:.2f} / 3")
    print("=" * 60)

    # ── SAVE RESULTS ─────────────────────────────────────────────────────────
    output = {
        "timestamp": datetime.now().isoformat(),
        "avg_retrieval_precision": round(avg_retrieval, 3),
        "avg_answer_score": round(avg_answer, 3),
        "results": results
    }
    output_path = Path("eval_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}\n")


if __name__ == "__main__":
    run_evaluation()