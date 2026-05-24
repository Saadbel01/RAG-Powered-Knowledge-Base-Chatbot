from langchain.prompts import ChatPromptTemplate


RAG_SYSTEM_PROMPT = """\
    You are a precise and helpful assistant.
    Answer the user's question using ONLY the context documents provided below.
    Rules:
    1. Use ONLY information present in the context.
    Never use outside knowledge.
    2. If the answer is not in the context, respond with exactly:
    "I do not have information about that in the provided documents."
    3. Never guess, invent, or assume facts not stated in the context.
    4. Be concise and direct. Avoid filler sentences.
    5. When relevant, cite the source (e.g. "According to manual.pdf,
    page 3...").
    Context documents:
    {context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "{question}"),
    ])
