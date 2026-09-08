"""Offline-first Zepto policy RAG service for Module 3."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
import requests
from fastapi import FastAPI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "zepto_policy_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"
POLICY_KEYWORDS = ("delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours")
GENERAL_REPLY = "I can only answer questions about Zepto policies right now."

# This is intentionally present as actual text for the optional MOCK_LLM=0 path.
STRUCTURED_PROMPT_TEMPLATE = """ROLE:
You are Zepto's policy support assistant.

CONTEXT:
{context}

TASK:
Answer the user's question using only the supplied context. Do not answer using information not present in the provided context.

FORMAT:
Return only valid JSON: {{"answer": "string", "sources": ["chunk id"], "confidence": 0.0}}.

LENGTH:
Use no more than 90 words.

FEW-SHOT EXAMPLE:
Context: doc_01: Standard delivery is free on orders over INR 149.
Question: When is standard delivery free?
JSON: {{"answer":"Standard delivery is free on orders over INR 149.","sources":["doc_01"],"confidence":1.0}}

Question: {query}
"""


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class AssistantState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved_chunks: list[dict[str, str]]
    answer: str
    sources: list[str]
    confidence: float


def mock_mode() -> bool:
    """MOCK_LLM is safe by default: unset and 1 both mean mock mode."""
    return os.getenv("MOCK_LLM", "1") != "0"


def read_documents() -> list[tuple[str, str]]:
    documents = []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        documents.append((path.stem, path.read_text(encoding="utf-8").strip()))
    if len(documents) != 8:
        raise RuntimeError(f"Expected 8 policy documents; found {len(documents)}.")
    return documents


@lru_cache(maxsize=1)
def service() -> tuple[SentenceTransformer, chromadb.Collection]:
    """Load MiniLM locally and upsert the eight corpus chunks into ChromaDB."""
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    documents = read_documents()
    ids, texts = zip(*documents)
    embeddings = model.encode(list(texts), normalize_embeddings=True).tolist()
    collection.upsert(ids=list(ids), documents=list(texts), metadatas=[{"document_id": doc_id} for doc_id in ids], embeddings=embeddings)
    return model, collection


def retrieve(query: str, limit: int = 3) -> list[dict[str, str]]:
    model, collection = service()
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=query_embedding, n_results=limit, include=["documents", "metadatas", "distances"])
    return [
        {"id": chunk_id, "text": text, "distance": str(distance)}
        for chunk_id, text, distance in zip(result["ids"][0], result["documents"][0], result["distances"][0])
    ]


def groq_text(prompt: str) -> str:
    """Optional real-LLM extension; never used unless MOCK_LLM=0 is explicit."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("MOCK_LLM=0 requires GROQ_API_KEY; use mock mode for the graded baseline.")
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), "messages": [{"role": "user", "content": prompt}], "temperature": 0},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def real_response(query: str, context: str, sources: list[str]) -> AskResponse:
    """Validate real LLM JSON and retry twice with a corrective instruction on failure."""
    prompt = STRUCTURED_PROMPT_TEMPLATE.format(query=query, context=context)
    last_error = "unknown validation failure"
    for attempt in range(3):  # original response plus at most two retries
        raw = groq_text(prompt)
        try:
            response = AskResponse.model_validate(json.loads(raw))
            return response.model_copy(update={"sources": sources or response.sources})
        except (json.JSONDecodeError, ValidationError) as error:
            last_error = str(error)
            prompt += "\nCORRECTION: Your previous output was invalid. Return only JSON with answer (string), sources (list), and confidence (0 to 1)."
    return AskResponse(answer=f"ERROR: real LLM output failed schema validation after 3 attempts ({last_error}).", sources=sources, confidence=0.0)


def classify_intent(state: AssistantState) -> AssistantState:
    query = state["query"]
    if mock_mode():
        intent: Literal["policy_question", "general_question"] = "policy_question" if any(keyword in query.lower() for keyword in POLICY_KEYWORDS) else "general_question"
    else:
        raw = groq_text(f"Classify this as exactly policy_question or general_question: {query}").strip().lower()
        intent = "policy_question" if "policy_question" in raw else "general_question"
    return {"intent": intent}


def retrieve_and_answer(state: AssistantState) -> AssistantState:
    chunks = retrieve(state["query"], limit=3)  # Real local embedding + cosine retrieval in both modes.
    source_ids = [chunk["id"] for chunk in chunks]
    if mock_mode():
        snippet = chunks[0]["text"][:200].replace("\n", " ")
        response = AskResponse(answer=f"Based on the retrieved context: {snippet}", sources=source_ids, confidence=1.0)
    else:
        context = "\n\n".join(f"{item['id']}: {item['text']}" for item in chunks)
        response = real_response(state["query"], context, source_ids)
    return {"retrieved_chunks": chunks, "answer": response.answer, "sources": response.sources, "confidence": response.confidence}


def direct_answer(state: AssistantState) -> AssistantState:
    if mock_mode():
        response = AskResponse(answer=GENERAL_REPLY, sources=[], confidence=1.0)
    else:
        response = real_response(state["query"], "No policy retrieval context was requested.", [])
    return {"answer": response.answer, "sources": response.sources, "confidence": response.confidence}


def route_after_intent(state: AssistantState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_graph():
    workflow = StateGraph(AssistantState)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer)
    workflow.add_node("direct_answer", direct_answer)
    workflow.add_edge(START, "classify_intent")
    workflow.add_conditional_edges("classify_intent", route_after_intent, {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"})
    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)
    return workflow.compile()


graph = build_graph()
app = FastAPI(title="Zepto Policy Support Assistant", version="1.0.0")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = graph.invoke({"query": request.query})
    return AskResponse(answer=result["answer"], sources=result["sources"], confidence=result["confidence"])


@app.get("/health")
def health() -> dict[str, object]:
    _, collection = service()
    return {"status": "ok", "mock_llm": mock_mode(), "collection": COLLECTION_NAME, "chunk_count": collection.count()}
