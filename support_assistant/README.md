# Module 3 — Zepto Policy Support Assistant

This is an offline-first Retrieval-Augmented Generation (RAG) service. The default `MOCK_LLM` mode makes no LLM-provider network calls and is the complete graded baseline.

## Run

From the repository root:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
$env:MOCK_LLM="1"     # optional: mock mode is already the default
Set-Location .\support_assistant
..\venv\Scripts\python.exe .\demo.py
..\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 7860
```

The first run downloads the open-source `all-MiniLM-L6-v2` model and stores its local cache. It then embeds all eight local documents into the persisted `chroma_db/` collection. No API key or LLM-provider call is needed in default mode.

In a second PowerShell terminal, test the running API:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:7860/ask -Method Post -ContentType "application/json" -Body '{"query":"What is the delivery fee for an order below INR 149?"}'
Invoke-RestMethod -Uri http://127.0.0.1:7860/ask -Method Post -ContentType "application/json" -Body '{"query":"Tell me a joke."}'
```

## Recorded mock-mode API examples

These demonstrate both LangGraph routes with `MOCK_LLM` unset/default. Exact wording is the deterministic mock response; source ordering is the Chroma top-3 similarity order.

```json
POST /ask {"query":"What is the delivery fee for an order below INR 149?"}
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.","sources":["doc_01","doc_03","doc_04"],"confidence":1.0}
```

```json
POST /ask {"query":"Tell me a joke."}
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

The policy question routes to `retrieve_and_answer`; its top result is `doc_01`, the Delivery Policy, which directly contains the delivery-fee rule. The unrelated question routes to `direct_answer`, performs no retrieval, and has no sources.

## Architecture: ingestion → embedding → retrieval → generation

```text
docs/doc_01.txt … docs/doc_08.txt
          │ ingestion: read_documents()
          ▼
all-MiniLM-L6-v2 local SentenceTransformer
          │ embedding: service()
          ▼
ChromaDB collection: zepto_policy_chunks
          │ retrieval: retrieve() / retrieve_and_answer node
          ▼
LangGraph: classify_intent → conditional route → retrieve_and_answer or direct_answer
          │ generation + Pydantic validation
          ▼
FastAPI POST /ask → {answer, sources, confidence}
```

`service()` ingests the eight exact policy files and embeds each whole document as one chunk with the local `all-MiniLM-L6-v2` model. It upserts those vectors into the persisted ChromaDB collection `zepto_policy_chunks`. The LangGraph `classify_intent` node routes policy-keyword queries to `retrieve_and_answer`, where `retrieve()` performs genuine top-3 cosine-similarity retrieval in both modes. General queries route to `direct_answer`.

`MOCK_LLM` branches only generation behavior: unset or `1` uses the required keyword heuristic and deterministic canned answers, with no LLM call. If explicitly set to `0`, each generation node uses the optional Groq-compatible real LLM path; `STRUCTURED_PROMPT_TEMPLATE` contains the role, context, task, JSON format, length limit, negative grounding rule, and few-shot example. `real_response()` validates the JSON Pydantic schema and retries up to two times with a corrective instruction on failure. The optional real path requires `GROQ_API_KEY` and is not needed for this project.

## Response schema

Every `POST /ask` response is validated by `AskResponse`:

```json
{"answer":"string","sources":["doc_01"],"confidence":1.0}
```

Mock policy answers receive the top three chunk IDs and confidence `1.0`; mock general answers receive `[]` and confidence `1.0`.

## Docker

Build from the repository root so the Dockerfile can copy the consolidated requirements file:

```powershell
docker build -f .\support_assistant\Dockerfile -t zepto-support-assistant .
docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant
```

Then send the same `POST /ask` request to `http://127.0.0.1:7860/ask`. The container uses the mock baseline by default and downloads the public open-source embedding model on its first startup.
