import json
import random
from pathlib import Path

import structlog
from langchain_groq import ChatGroq

from rag_chatbot.config import get_settings
from rag_chatbot.ingestion.chunker import create_chunks
from rag_chatbot.ingestion.loader import load_documents

log = structlog.get_logger(__name__)


def generate_testset(n_samples: int = 25) -> list[dict]:

    results: list[dict] = []

    settings = get_settings()

    documents = load_documents(settings.data_folder)

    chunks = create_chunks(documents)

    if len(chunks) < n_samples:
        n_samples = len(chunks)

    random_samples = random.sample(chunks, n_samples)

    model = ChatGroq(
        model=settings.groq_model,
        temperature=0,
        groq_api_key=settings.groq_api_key)

    for i, sample in enumerate(random_samples):

        try:
            json_format = '{"question": "...", "answer": "..."}'
            prompt = f"""Read the following text and generate ONE clear
                    question that can be
                    answered ONLY from this text,
                    plus the exact correct answer.
                    Respond in JSON: {json_format}

                    Text:
                    {sample.page_content}"""
            response = json.loads(model.invoke(prompt).content)

            results.append({
                "question": response["question"],
                "ground_truth": response["answer"],
                "context": sample.page_content
            })
            log.info("generate.progress", done=i + 1, total=n_samples)
        except Exception as e:
            log.error("generate.error", chunk_id=i, error=str(e))

    output_path = Path("evaluation/testset.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("generate.done", saved=str(output_path), samples=len(results))
    return results


if __name__ == "__main__":
    generate_testset()
