#!/usr/bin/env python3
"""Core logic of the arXiv digest explorer: retrieval, scoring, storage."""

import datetime
import html
import json
import logging
import os
import pickle
import re
import sqlite3


DB_PATH = os.environ.get("ARXIV_DIGEST_DB", "reading_list.db")
POSITIVE_LOG = "positive.jsonl"
NEGATIVE_LOG = "negative.jsonl"

# Papers nobody touched for this long silently disappear from the reading list.
EXPIRE_DAYS = 4 * 30

KEYWORDS = [
    "translation", "multilingual", "cross-lingual", "multimodal",
    "tokenization", "language-vision", "vision-language"]

RE_KEYWORDS = [re.compile(kwrd, re.IGNORECASE) for kwrd in KEYWORDS]


# ---------------------------------------------------------------- retrieval

def retrieve_recent(start_date, page_size=200):
    """Download cs.CL pre-prints updated after `start_date` (oldest first).

    The arXiv API cannot filter by update date, so we walk the pre-prints
    from the newest one and stop at the first one we have already seen.
    The results are paged lazily, which means we transfer at most one page
    more than there are new pre-prints.
    """
    import arxiv

    client = arxiv.Client(
        page_size=page_size,
        num_retries=10,
        delay_seconds=3.0,
    )
    search = arxiv.Search(
        query="cs.CL",
        max_results=None,  # we stop ourselves once we reach `start_date`
        sort_by=arxiv.SortCriterion.LastUpdatedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    found_recent = []
    for result in client.results(search):
        if result.updated <= start_date:
            break
        found_recent.append(result)
        if len(found_recent) % page_size == 0:
            logging.info("Retrieved %d new pre-prints so far.",
                         len(found_recent))
    else:
        logging.warning("Ran out of arXiv results before reaching %s.",
                        start_date)

    output = []
    for result in found_recent:
        output.append({
            "title": result.title,
            "authors": ", ".join(aut.name for aut in result.authors),
            "arxiv_id": result.entry_id.split("/")[-1],
            "abstract": result.summary,
            "date": result.updated.isoformat(),
            "url": result.entry_id.replace("http", "https"),
            "comment": result.comment,
        })

    output.reverse()
    return output


# ------------------------------------------------------------------ scoring

class Scorer:
    """Transformer classifier assigning a probability of me reading a paper."""

    def __init__(self, model_path, batch_size=8, max_length=512):
        logging.info("Import Transformers and Torch.")
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification)
        logging.info("Load the scoring model from '%s'.", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path)
        self.model.eval()
        self.batch_size = batch_size
        self.max_length = max_length
        logging.info("Loaded.")

    def score(self, items):
        """Add a "score" field to every item."""
        import torch

        # Similar lengths in one batch keeps the padding, and with it the
        # peak memory, down.
        order = sorted(range(len(items)),
                       key=lambda i: len(items[i]["abstract"]))
        for start in range(0, len(order), self.batch_size):
            batch = [items[i] for i in order[start:start + self.batch_size]]
            prompts = [
                "Title: {}. Abstract: {}.".format(
                    item["title"], item["abstract"].replace("\n", " "))
                for item in batch]
            tokenized = self.tokenizer(
                prompts, return_tensors="pt", truncation=True,
                max_length=self.max_length, padding=True)
            with torch.inference_mode():
                logits = self.model(**tokenized)[0]
            scores = torch.softmax(logits, dim=1)[:, 1]
            for item, score in zip(batch, scores.tolist()):
                item["score"] = score
            del tokenized, logits, scores
        return items


# ------------------------------------------------------------- highlighting

class Highlighter:
    """Renders an abstract as HTML with keywords and TF-IDF terms marked."""

    def __init__(self, tfidf_path=None, top_n=10):
        self.vectorizer = None
        self.top_n = top_n
        if tfidf_path is not None:
            logging.info("Load the TF-IDF vectorizer from '%s'.", tfidf_path)
            with open(tfidf_path, "rb") as f_tfidf:
                self.vectorizer = pickle.load(f_tfidf)
            logging.info("Loaded.")

    def _tfidf_words(self, text):
        if self.vectorizer is None:
            return []
        tfidf = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        topn_ids = tfidf.indices[tfidf.data.argsort()[-self.top_n:]]
        return [feature_names[i] for i in topn_ids]

    def to_html(self, text, with_tfidf=True):
        spans = []
        for regex in RE_KEYWORDS:
            for match in regex.finditer(text):
                spans.append((match.start(), match.end(), "keyword"))
        if with_tfidf:
            for word in self._tfidf_words(text):
                regex = re.compile(r"\b" + re.escape(word) + r"\b",
                                   re.IGNORECASE)
                for match in regex.finditer(text):
                    spans.append((match.start(), match.end(), "tfidf"))

        # Keywords win over TF-IDF terms, longer matches over shorter ones.
        spans.sort(key=lambda s: (s[0], s[2] != "keyword", -s[1]))

        parts = []
        position = 0
        for start, end, kind in spans:
            if start < position:  # overlapping match, already highlighted
                continue
            parts.append(html.escape(text[position:start]))
            parts.append(
                f'<mark class="{kind}">{html.escape(text[start:end])}</mark>')
            position = end
        parts.append(html.escape(text[position:]))
        return "".join(parts)


# ------------------------------------------------------------------ storage

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id   TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    authors    TEXT NOT NULL,
    abstract   TEXT NOT NULL,
    comment    TEXT,
    url        TEXT NOT NULL,
    date       TEXT NOT NULL,
    score      REAL,
    first_seen TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'new',
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS papers_status_score
    ON papers (status, score DESC);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect(db_path=DB_PATH):
    connection = sqlite3.connect(db_path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    # No WAL here: its shared-memory file is mmap'd, which SIGBUSes on the
    # network filesystems this repo tends to live on.
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.executescript(SCHEMA)
    migrate(connection)
    return connection


def migrate(connection):
    """Make `score` nullable in databases written by the first version."""
    columns = {row["name"]: row
               for row in connection.execute("PRAGMA table_info(papers)")}
    if not columns["score"]["notnull"]:
        return
    logging.info("Migrating the database to a nullable score.")
    connection.executescript("""
        DROP INDEX IF EXISTS papers_status_score;
        ALTER TABLE papers RENAME TO papers_old;
    """ + SCHEMA + """
        INSERT INTO papers SELECT * FROM papers_old;
        DROP TABLE papers_old;
    """)
    connection.commit()


def get_last_date(connection):
    """The newest paper date seen so far, seeded from the old `last_date`."""
    row = connection.execute(
        "SELECT value FROM meta WHERE key = 'last_date'").fetchone()
    if row is not None:
        return row["value"]
    if os.path.exists("last_date"):
        with open("last_date") as f_date:
            return f_date.read().strip()
    return None


def set_last_date(connection, value):
    connection.execute(
        "INSERT INTO meta (key, value) VALUES ('last_date', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (value,))
    connection.commit()


def add_papers(connection, items, status="new"):
    """Store downloaded papers, skipping the ones we already know."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    added = 0
    for item in items:
        cursor = connection.execute(
            "INSERT OR IGNORE INTO papers "
            "(arxiv_id, title, authors, abstract, comment, url, date, score, "
            " first_seen, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (item["arxiv_id"], item["title"], item["authors"],
             item["abstract"], item.get("comment"), item["url"], item["date"],
             item.get("score"), now, status))
        added += cursor.rowcount
    connection.commit()
    return added


def expire_old(connection, days=EXPIRE_DAYS):
    """Drop untouched papers older than `days`; nothing gets logged."""
    cutoff = (datetime.datetime.now(datetime.timezone.utc)
              - datetime.timedelta(days=days)).isoformat()
    cursor = connection.execute(
        "UPDATE papers SET status = 'expired', decided_at = ? "
        "WHERE status = 'pending' AND first_seen < ?",
        (datetime.datetime.now(datetime.timezone.utc).isoformat(), cutoff))
    connection.commit()
    return cursor.rowcount


def pending_papers(connection):
    """The reading list, best-scoring paper first."""
    return connection.execute(
        "SELECT * FROM papers WHERE status = 'pending' "
        "ORDER BY score DESC, date DESC").fetchall()


def counts(connection):
    rows = connection.execute(
        "SELECT status, COUNT(*) AS count FROM papers GROUP BY status")
    return {row["status"]: row["count"] for row in rows}


def decide(connection, arxiv_id, decision):
    """Record a yes/no decision and append the paper to the training log."""
    assert decision in ("yes", "no")
    row = connection.execute(
        "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
    if row is None or row["status"] != "pending":
        return None

    connection.execute(
        "UPDATE papers SET status = ?, decided_at = ? WHERE arxiv_id = ?",
        (decision, datetime.datetime.now(datetime.timezone.utc).isoformat(),
         arxiv_id))
    connection.commit()

    log_path = POSITIVE_LOG if decision == "yes" else NEGATIVE_LOG
    item = {
        "arxiv_id": row["arxiv_id"],
        "date": row["date"],
        "title": row["title"],
        "authors": row["authors"],
        "url": row["url"],
        "abstract": row["abstract"],
        "comment": row["comment"],
        "score": row["score"],
    }
    with open(log_path, "a") as f_log:
        print(json.dumps(item), file=f_log)
    return row


def parse_date(text):
    """ISO first, so that only human input like "01.08.2026" is day-first."""
    try:
        return datetime.datetime.fromisoformat(text)
    except ValueError:
        from dateutil.parser import parse as date_parse
        return date_parse(text, dayfirst=True)


def fetch_new(connection, start_date=None):
    """Download pre-prints newer than the last one we stored.

    They are saved unscored and `last_date` moves right away, so that a
    later crash never makes us download the same abstracts again.
    """
    stored = get_last_date(connection)
    if start_date is not None and stored is not None:
        logging.warning(
            "--start-date overrides the stored date %s, so everything since "
            "%s is downloaded again. Drop it to continue where the last run "
            "stopped.", stored, start_date)
    if start_date is None:
        start_date = stored
    if start_date is None:
        raise ValueError(
            "No start date known: pass --start-date for the first run.")
    if isinstance(start_date, str):
        start_date = parse_date(start_date)
    if start_date.tzinfo is None:  # arXiv dates are always aware
        start_date = start_date.replace(tzinfo=datetime.timezone.utc)

    logging.info("The youngest stored paper was %s.", start_date)
    logging.info("Retrieving abstracts from arXiv.")
    items = retrieve_recent(start_date)
    logging.info("Downloading finished, retrieved %d abstracts.", len(items))
    if not items:
        return 0

    added = add_papers(connection, items)
    set_last_date(connection, max(item["date"] for item in items))
    logging.info("Stored %d new abstracts, waiting to be scored.", added)
    return added


def unscored_count(connection):
    return connection.execute(
        "SELECT COUNT(*) AS count FROM papers WHERE status = 'new'"
    ).fetchone()["count"]


def score_new(connection, scorer, threshold=0.01, chunk_size=64):
    """Score the stored abstracts that have no score yet.

    Every chunk is committed on its own, so an interrupted run picks up
    where it stopped instead of scoring everything again.
    """
    total = unscored_count(connection)
    if not total:
        return {"scored": 0, "kept": 0}
    logging.info("Scoring %d abstracts.", total)

    scored = 0
    kept = 0
    while True:
        rows = connection.execute(
            "SELECT arxiv_id, title, abstract FROM papers "
            "WHERE status = 'new' ORDER BY date LIMIT ?",
            (chunk_size,)).fetchall()
        if not rows:
            break

        items = [{"arxiv_id": row["arxiv_id"], "title": row["title"],
                  "abstract": row["abstract"]} for row in rows]
        scorer.score(items)
        for item in items:
            status = "pending" if item["score"] >= threshold else "rejected"
            kept += status == "pending"
            connection.execute(
                "UPDATE papers SET score = ?, status = ? WHERE arxiv_id = ?",
                (item["score"], status, item["arxiv_id"]))
        connection.commit()

        scored += len(items)
        logging.info("Scored %d of %d abstracts, %d kept.", scored, total, kept)

    return {"scored": scored, "kept": kept}


def fetch_and_score(connection, scorer=None, threshold=0.01, start_date=None,
                    chunk_size=64):
    """Download new pre-prints and score everything that has no score yet."""
    added = fetch_new(connection, start_date)
    if scorer is None:
        # Without a model nothing can be ranked, so keep everything.
        connection.execute(
            "UPDATE papers SET score = 0.0, status = 'pending' "
            "WHERE status = 'new'")
        connection.commit()
        result = {"scored": 0, "kept": added}
    else:
        result = score_new(connection, scorer, threshold, chunk_size)
    expired = expire_old(connection)
    logging.info(
        "Added %d papers to the reading list, %d expired.",
        result["kept"], expired)
    return {"retrieved": added, "added": result["kept"], "expired": expired}
