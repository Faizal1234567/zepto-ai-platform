"""Build the Module 1 book catalogue data pipeline from scratch.

The script scrapes three Books to Scrape categories, cleans and enriches the
records, loads a normalized SQLite database, and saves reproducible SQL and
pandas query evidence under ``data_pipeline/output``.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50
TARGET_CATEGORY_COUNT = 3
REQUEST_TIMEOUT_SECONDS = 30

MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = MODULE_DIR / "output"
DATABASE_PATH = MODULE_DIR / "books_catalogue.sqlite"
CSV_PATH = OUTPUT_DIR / "books_cleaned.csv"
QUERY_OUTPUT_PATH = OUTPUT_DIR / "query_results.md"
SUMMARY_PATH = OUTPUT_DIR / "run_summary.json"

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def fetch_soup(session: requests.Session, url: str) -> BeautifulSoup:
    """Fetch one HTML page and fail clearly on an HTTP error."""
    response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def category_urls(session: requests.Session) -> list[tuple[str, str]]:
    """Return the first three catalogue category names and their URLs."""
    soup = fetch_soup(session, BASE_URL)
    links = soup.select("ul.nav-list ul li a")[:TARGET_CATEGORY_COUNT]
    if len(links) < TARGET_CATEGORY_COUNT:
        raise RuntimeError("The catalogue did not expose three category links.")
    return [
        (link.get_text(strip=True), urljoin(BASE_URL, link["href"])) for link in links
    ]


def scrape_category(
    session: requests.Session, category_name: str, category_url: str
) -> list[dict[str, str]]:
    """Scrape every listing page in one category."""
    rows: list[dict[str, str]] = []
    next_url: str | None = category_url
    while next_url:
        soup = fetch_soup(session, next_url)
        for book in soup.select("article.product_pod"):
            rating_classes = book.select_one("p.star-rating").get("class", [])
            rating_text = next(
                (name for name in rating_classes if name in RATING_MAP), ""
            )
            rows.append(
                {
                    "title": book.select_one("h3 a").get("title", "").strip(),
                    "price_raw": book.select_one("p.price_color").get_text(strip=True),
                    "star_rating_raw": rating_text,
                    "availability_raw": book.select_one("p.instock.availability").get_text(
                        " ", strip=True
                    ),
                    "category": category_name,
                }
            )
        next_link = soup.select_one("li.next a")
        next_url = urljoin(next_url, next_link["href"]) if next_link else None
    return rows


def parse_price(value: object) -> float | None:
    """Extract a numeric GBP price from site text such as '£51.77'."""
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def parse_stock(value: object) -> bool | None:
    text = " ".join(str(value).split()).lower()
    if "in stock" in text:
        return True
    if "out of stock" in text:
        return False
    return None


def clean_books(raw_rows: Iterable[dict[str, str]]) -> pd.DataFrame:
    """Clean scraped records and apply the documented malformed-data policy.

    Missing/malformed numeric price or rating values are median-imputed. Rows
    missing a title, category, or recognizable stock state are dropped because
    those values are identifiers/categorical facts and cannot be inferred.
    """
    frame = pd.DataFrame(raw_rows)
    if frame.empty:
        raise RuntimeError("No books were scraped; database was not created.")

    frame["price_gbp"] = frame["price_raw"].map(parse_price)
    frame["rating"] = frame["star_rating_raw"].map(RATING_MAP)
    frame["in_stock"] = frame["availability_raw"].map(parse_stock)

    for column in ("price_gbp", "rating"):
        median = frame[column].median()
        if pd.isna(median):
            raise RuntimeError(f"Cannot impute {column}: every value failed parsing.")
        frame[column] = frame[column].fillna(median)

    frame["title"] = frame["title"].astype("string").str.strip()
    frame["category"] = frame["category"].astype("string").str.strip()
    frame = frame.dropna(subset=["title", "category", "in_stock"])
    frame = frame[(frame["title"] != "") & (frame["category"] != "")].copy()

    frame["price_gbp"] = frame["price_gbp"].astype(float)
    frame["rating"] = frame["rating"].round().astype(int)
    frame["in_stock"] = frame["in_stock"].astype(bool)
    frame["price_inr"] = (frame["price_gbp"] * GBP_TO_INR).round(2)
    return frame[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]


def create_database(cleaned: pd.DataFrame, database_path: Path) -> None:
    """Create the normalized categories/books SQLite schema and load its rows."""
    if database_path.exists():
        database_path.unlink()
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT NOT NULL UNIQUE
            );
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
                category_id INTEGER NOT NULL,
                FOREIGN KEY (category_id) REFERENCES categories(category_id)
            );
            """
        )
        categories = sorted(cleaned["category"].unique())
        connection.executemany(
            "INSERT INTO categories (category_name) VALUES (?)", [(name,) for name in categories]
        )
        category_ids = dict(
            connection.execute("SELECT category_name, category_id FROM categories").fetchall()
        )
        book_rows = [
            (
                row.title,
                row.price_gbp,
                row.price_inr,
                row.rating,
                int(row.in_stock),
                category_ids[row.category],
            )
            for row in cleaned.itertuples(index=False)
        ]
        connection.executemany(
            """INSERT INTO books
               (title, price_gbp, price_inr, rating, in_stock, category_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            book_rows,
        )


QUERIES = {
    "1. SELECT + WHERE (in-stock books rated four or five)": """
        SELECT title, rating, price_gbp
        FROM books
        WHERE in_stock = 1 AND rating >= 4
        ORDER BY rating DESC, price_gbp DESC
        LIMIT 10;
    """,
    "2. ORDER BY + LIMIT (five most expensive books)": """
        SELECT title, price_gbp, price_inr
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 5;
    """,
    "3. DISTINCT (categories represented)": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
    """,
    "4. BETWEEN (mid-priced books)": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE price_gbp BETWEEN 20 AND 30
        ORDER BY price_gbp;
    """,
    "5. IN (selected ratings)": """
        SELECT title, rating, in_stock
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title
        LIMIT 15;
    """,
    "6. JOIN (ten highest-rated books with category)": """
        SELECT b.title, c.category_name AS category, b.rating, b.price_gbp, b.price_inr, b.in_stock
        FROM books AS b
        JOIN categories AS c ON b.category_id = c.category_id
        ORDER BY b.rating DESC, b.price_gbp DESC, b.title ASC
        LIMIT 10;
    """,
}


def dataframe_markdown(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as Markdown without an extra tabulate dependency."""
    columns = [str(column) for column in frame.columns]

    def display(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    rows.extend("| " + " | ".join(display(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def save_query_evidence(database_path: Path) -> tuple[bool, dict[str, int]]:
    """Execute required SQL, save printed output, and prove pandas merge matches JOIN."""
    with sqlite3.connect(database_path) as connection:
        # Use pandas.read_sql exactly as required, for every stored SQL result.
        results = {name: pd.read_sql(sql, connection) for name, sql in QUERIES.items()}
        books_df = pd.read_sql("SELECT * FROM books", connection)
        categories_df = pd.read_sql("SELECT * FROM categories", connection)

    join_sql = results["6. JOIN (ten highest-rated books with category)"]
    join_merge = (
        books_df.merge(categories_df, on="category_id", how="inner")
        .rename(columns={"category_name": "category"})[
            ["title", "category", "rating", "price_gbp", "price_inr", "in_stock"]
        ]
        .sort_values(["rating", "price_gbp", "title"], ascending=[False, False, True])
        .head(10)
        .reset_index(drop=True)
    )
    join_sql = join_sql.reset_index(drop=True)
    join_matches_merge = join_sql.equals(join_merge)

    sections = [
        "# Executed SQL query results\n",
        "Generated by `python pipeline.py`. This file is evidence, not hand-entered output.\n",
    ]
    for name, sql in QUERIES.items():
        sections.extend([f"## {name}\n", "```sql", sql.strip(), "```\n", dataframe_markdown(results[name]), "\n"])
    sections.extend(
        [
            "## pandas read_sql and merge verification\n",
            "The JOIN result was read with `pd.read_sql`; the five-most-expensive query was also read with pandas. "
            f"The in-memory `pd.merge` reproduction matches the SQL JOIN: **{join_matches_merge}**.\n",
            "### JOIN result read with pandas\n",
            dataframe_markdown(join_sql),
            "\n### Same result reproduced with pandas.merge\n",
            dataframe_markdown(join_merge),
            "",
        ]
    )
    QUERY_OUTPUT_PATH.write_text("\n".join(sections), encoding="utf-8")
    return join_matches_merge, {name: len(frame) for name, frame in results.items()}


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Zepto-AI-Platform-Student-Project/1.0"})
        selected_categories = category_urls(session)
        raw_rows = [
            row
            for name, url in selected_categories
            for row in scrape_category(session, name, url)
        ]

    cleaned = clean_books(raw_rows)
    if len(cleaned) < 60 or cleaned["category"].nunique() < 3:
        raise RuntimeError(
            f"Expected at least 60 books across 3 categories; got {len(cleaned)} books "
            f"across {cleaned['category'].nunique()} categories."
        )
    cleaned.to_csv(CSV_PATH, index=False)
    create_database(cleaned, DATABASE_PATH)
    join_matches_merge, query_row_counts = save_query_evidence(DATABASE_PATH)
    summary = {
        "books_scraped": len(raw_rows),
        "books_loaded": len(cleaned),
        "categories": sorted(cleaned["category"].unique().tolist()),
        "currency_conversion": {"gbp_to_inr": GBP_TO_INR},
        "join_matches_pandas_merge": join_matches_merge,
        "query_result_row_counts": query_row_counts,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nCreated {DATABASE_PATH.name} and evidence in {OUTPUT_DIR.relative_to(MODULE_DIR)}.")


if __name__ == "__main__":
    main()
