# Zepto Data & AI Platform

One repository containing three connected capstone modules: a catalogue data pipeline, an analytics pipeline, and a grounded support assistant. The repository uses one consolidated `requirements.txt` at its root.

## Module 1: data pipeline

Module 1 is implemented in [`data_pipeline`](data_pipeline/README.md). It scrapes Books to Scrape, cleans and converts the data, creates a normalized SQLite database, and saves reproducible SQL/pandas query evidence.

Run it from the repository root:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe .\data_pipeline\pipeline.py
```

Modules 2 and 3 are intentionally not implemented as part of the current Module 1 delivery.
