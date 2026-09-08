# Module 1 — Data Pipeline

This module turns live book-catalogue pages from [Books to Scrape](https://books.toscrape.com/) into a clean relational SQLite database. The product domain is books, but the workflow—scrape, validate, enrich, store, and query—is directly applicable to catalogue pricing data.

## Run

From the repository root, install the consolidated dependencies and run the pipeline:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe .\data_pipeline\pipeline.py
```

The script makes HTTP requests to the public practice site, so an internet connection is required only while rebuilding the data. It scrapes every page of the first three listed categories (Travel, Mystery, and Historical Fiction at the time of writing) and verifies that the cleaned result has at least 60 books across at least three categories.

## Outputs

- `books_catalogue.sqlite` — normalized SQLite database, regenerated from scratch by `pipeline.py`.
- `output/books_cleaned.csv` — cleaned dataset with the final typed columns.
- `output/query_results.md` — six executed SQL statements and their real outputs, followed by pandas verification.
- `output/run_summary.json` — record count, category coverage, fixed conversion rate, and SQL/pandas JOIN equality result.

## Data cleaning and enrichment

`price_gbp` is extracted from the site’s GBP price text and converted to `float`. Text ratings (`One` through `Five`) are mapped to integer `rating` values 1–5. Availability text is normalized to boolean `in_stock`.

If a price or rating cannot be parsed, the pipeline median-imputes that numeric field. If title, category, or availability cannot be parsed, it drops the row because these categorical/identity values cannot be credibly inferred. This makes malformed input explicit without stopping a complete pipeline run.

`price_inr` is calculated using the required project baseline of **1 GBP = 105.50 INR**. This is a fixed assignment constant, not a live exchange rate, so no currency API is called.

## Database design

The schema is normalized into two related tables:

```text
categories
  category_id  INTEGER PRIMARY KEY
  category_name TEXT UNIQUE

books
  book_id      INTEGER PRIMARY KEY
  title        TEXT
  price_gbp    REAL
  price_inr    REAL
  rating       INTEGER (1–5)
  in_stock     INTEGER (0/1)
  category_id  INTEGER FOREIGN KEY → categories.category_id
```

The generated query evidence includes SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, IN, and a categories/books JOIN. It reads SQL results back with `pd.read_sql` and independently recreates the JOIN with `pd.merge`, then records whether the two DataFrames are identical.
