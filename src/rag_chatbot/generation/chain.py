import structlog
import time
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential

from rag_chatbot.config import get_settings
from rag_chatbot.generation.prompts import RAG_PROMPT


log = structlog.get_logger(__name__)


def format_docs(docs: list[Document]) -> str:

    if not docs:
        return "No relevant documents were retrieved for this query."

    parts: list[str] = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown source")

        page = doc.metadata.get("page", "")

        ref = (f"{source}, p.{page}" if page != "" else source)

        parts.append(
            f"[Document {i} | {ref}]\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(parts)


def create_llm() -> ChatGroq:

    cfg = get_settings()

    return ChatGroq(
        model=cfg.groq_model,
        temperature=0,
        max_tokens=1024,
        groq_api_key=cfg.groq_api_key,
    )


def create_rag_chain(retriever):

    llm = create_llm()

    return (
        {
            "context":
                retriever | format_docs,

            "question":
                RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


@retry(stop=stop_after_attempt(2), wait=wait_exponential(
        multiplier=1, min=1, max=4))
def ask_with_retry(chain, question: str) -> dict:

    start = time.perf_counter()

    answer = chain.invoke(question)

    latency_ms = int((time.perf_counter() - start) * 1000)

    log.info("chain.ok", latency_ms=latency_ms, question_preview=question[:60])

    return {
        "answer": answer,
        "latency_ms": latency_ms,
        "success": True
    }
