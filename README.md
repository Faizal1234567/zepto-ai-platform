# Zepto Data & AI Platform

One repository containing three connected capstone modules: a catalogue data pipeline, an analytics pipeline, and a grounded support assistant. The repository uses one consolidated `requirements.txt` at its root.

## Module 1: data pipeline

Module 1 is implemented in [`data_pipeline`](data_pipeline/README.md). It scrapes Books to Scrape, cleans and converts the data, creates a normalized SQLite database, and saves reproducible SQL/pandas query evidence.

Run it from the repository root:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe .\data_pipeline\pipeline.py
```

## Module 2: analytics

Module 2 is in [`analytics`](analytics/README.md). Run `01_eda.py` once to load and profile Titanic, then run `02_modeling.py` to train and evaluate the models. Detailed instructions are in the module README.

## Module 3: support assistant

Module 3 is in [`support_assistant`](support_assistant/README.md). It provides an offline-first LangGraph and FastAPI policy assistant backed by local MiniLM embeddings and ChromaDB. Its default mock mode needs no API key or LLM-provider call.
