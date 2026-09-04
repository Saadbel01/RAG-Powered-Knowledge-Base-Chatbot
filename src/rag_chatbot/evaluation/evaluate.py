import json
from pathlib import Path

import structlog
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from rag_chatbot.config import get_settings
from rag_chatbot.generation.chain import create_rag_chain
from rag_chatbot.retrieval.retriever import HybridRetriever

log = structlog.get_logger(__name__)


TARGETS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.70,
}


def run_evaluation(testset_path: str = "evaluation/testset.json") -> dict:

    settings = get_settings()

    data = json.loads(Path(testset_path).read_text(encoding="utf-8"))

    log.info("eval.start", samples=len(data))

    retriever = HybridRetriever()
    chain = create_rag_chain(retriever)

    rows: list[dict] = []

    for item in data:
        q = item["question"]

        chunks = retriever.retrieve(q)
        answer = chain.invoke(q)

        rows.append(
            {
                "question": q,
                "answer": answer,
                "contexts": [c.page_content for c in chunks],
                "ground_truth": item["ground_truth"],
            }
        )

    dataset = Dataset.from_list(rows)

    judge = ChatGroq(
        model=settings.groq_model,
        temperature=0,
        groq_api_key=settings.groq_api_key,
    )

    embedder = HuggingFaceEmbeddings(
        model_name=settings.embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=judge,
        embeddings=embedder,
    )

    df = scores.to_pandas()

    means = df[list(TARGETS.keys())].mean().to_dict()

    print("\n" + "=" * 58)
    print(" RAGAS EVALUATION RESULTS")
    print("=" * 58)

    header = f" {'Metric':<24} {'Score':>6} {'Target':>6} Status"
    print(header)
    print(" " + "-" * 50)

    for metric, score in means.items():
        bar = "#" * int(score * 20)
        status = "PASS" if score >= TARGETS[metric] else "FAIL"

        print(
            f" {metric:<24} "
            f"{score:.3f} "
            f"{TARGETS[metric]:.3f} "
            f"{status}"
        )

        print(f" {'':24} [{bar:<20}]")

    print("=" * 58)

    out = Path("evaluation/ragas_results.csv")
    df.to_csv(out, index=False)

    log.info("eval.done", csv=str(out))

    return means


if __name__ == "__main__":
    run_evaluation()
