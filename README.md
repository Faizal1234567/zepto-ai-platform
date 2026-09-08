# Zepto Data & AI Platform

One repository containing three connected capstone modules: a catalogue data pipeline, an analytics pipeline, and a grounded support assistant.

## Setup

The root `requirements.txt` is the consolidated development environment for all three modules. Create and activate a virtual environment, then install it from the repository root:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`support_assistant/requirements.txt` is the small runtime dependency list used by its Docker image. It is not required when the root consolidated environment has already been installed.

## Module 1: data pipeline

Module 1 is implemented in [`data_pipeline`](data_pipeline/README.md). It scrapes Books to Scrape, parses and validates catalogue fields, converts GBP to INR using the assignment's fixed 105.50 conversion rate, then loads a normalized SQLite categories/books database. SQL evidence and a pandas merge cross-check make the result reproducible.

Run it from the repository root:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe .\data_pipeline\pipeline.py
```

## Module 2: analytics

Module 2 is in [`analytics`](analytics/README.md). Run `01_eda.py` once to load and profile Titanic, then run `02_modeling.py` to train and evaluate the models. Detailed instructions are in the module README.

```powershell
.\venv\Scripts\python.exe .\analytics\01_eda.py
.\venv\Scripts\python.exe .\analytics\02_modeling.py
```

The first stage makes the module's one Seaborn dataset load and saves `analytics/titanic.csv` as the committed offline fallback. The second stage reads that CSV, uses train-only preprocessing within scikit-learn pipelines, evaluates/tunes the models, and saves the complete best classifier pipeline.

## Module 3: support assistant

Module 3 is in [`support_assistant`](support_assistant/README.md). It provides an offline-first LangGraph and FastAPI policy assistant backed by local MiniLM embeddings and ChromaDB. Its default mock mode needs no API key or LLM-provider call.

```powershell
Set-Location .\support_assistant
$env:MOCK_LLM="1"
..\venv\Scripts\python.exe .\demo.py
..\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 7860
```

Test the running API in a second terminal with a `POST` request to `http://127.0.0.1:7860/ask`, or open `http://127.0.0.1:7860/docs` for the interactive API documentation. The service deterministically routes policy questions to real local ChromaDB retrieval and returns validated JSON; unrelated questions receive a fixed policy-only response.

To build and run the same service in Docker from the repository root:

```powershell
docker build -f .\support_assistant\Dockerfile -t zepto-support-assistant .
docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant
```
