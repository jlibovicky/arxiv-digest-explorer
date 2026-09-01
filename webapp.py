#!/usr/bin/env python3
"""Web interface for exploring the arXiv reading list ordered by score."""

import argparse
import logging
import threading
import webbrowser

from flask import Flask, jsonify, render_template, request

import digest_core
import zotero_reminder


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

QUEUE_PAGE_SIZE = 30
ZOTERO_SAMPLE = 3


app = Flask(__name__)
app.config["state"] = None


class State:
    """Everything the request handlers need, shared between threads."""

    def __init__(self, connection, db_path, highlighter, scorer, threshold,
                 chunk_size=64):
        self.connection = connection
        self.db_path = db_path
        self.highlighter = highlighter
        self.scorer = scorer
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.lock = threading.Lock()
        self.refreshing = False
        self.last_refresh = None

    def refresh(self):
        """Download and score new pre-prints. Safe to call from a thread."""
        with self.lock:
            if self.refreshing:
                return None
            self.refreshing = True
        # A separate connection so that serving the list is not blocked
        # while arXiv is being downloaded.
        connection = digest_core.connect(self.db_path)
        try:
            result = digest_core.fetch_and_score(
                connection, self.scorer, self.threshold, None, self.chunk_size)
            self.last_refresh = result
            return result
        except Exception as exc:  # keep the server alive on arXiv hiccups
            logging.exception("Refresh failed.")
            self.last_refresh = {"error": str(exc)}
            return self.last_refresh
        finally:
            connection.close()
            self.refreshing = False


def state():
    return app.config["state"]


def paper_to_json(row, highlighter):
    return {
        "arxiv_id": row["arxiv_id"],
        "title": row["title"],
        "title_html": highlighter.to_html(row["title"], with_tfidf=False),
        "authors": row["authors"],
        "comment": row["comment"],
        "url": row["url"],
        "date": row["date"],
        "score": row["score"],
        "abstract_html": highlighter.to_html(row["abstract"]),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/queue")
def api_queue():
    """A page of the reading list, best-scoring paper first."""
    current = state()
    limit = min(int(request.args.get("limit", QUEUE_PAGE_SIZE)), 100)
    cursor_score = request.args.get("cursor_score", type=float)
    cursor_date = request.args.get("cursor_date")

    query = "SELECT * FROM papers WHERE status = 'pending'"
    params = []
    if cursor_score is not None and cursor_date is not None:
        query += " AND (score < ? OR (score = ? AND date < ?))"
        params += [cursor_score, cursor_score, cursor_date]
    query += " ORDER BY score DESC, date DESC LIMIT ?"
    params.append(limit)

    with current.lock:
        rows = current.connection.execute(query, params).fetchall()
        stats = digest_core.counts(current.connection)
        remaining = stats.get("pending", 0)

    papers = [paper_to_json(row, current.highlighter) for row in rows]
    return jsonify({"papers": papers, "pending": remaining})


@app.route("/api/decide", methods=["POST"])
def api_decide():
    current = state()
    payload = request.get_json(force=True)
    arxiv_id = payload.get("arxiv_id")
    decision = payload.get("decision")
    if decision not in ("yes", "no"):
        return jsonify({"error": "decision must be 'yes' or 'no'"}), 400

    with current.lock:
        row = digest_core.decide(current.connection, arxiv_id, decision)
    if row is None:
        return jsonify({"error": "unknown or already decided paper"}), 404
    return jsonify({"status": "ok"})


@app.route("/api/status")
def api_status():
    current = state()
    with current.lock:
        stats = digest_core.counts(current.connection)
    return jsonify({
        "refreshing": current.refreshing,
        "unscored": stats.get("new", 0),
        "last_refresh": current.last_refresh,
        "counts": stats,
    })


@app.route("/api/zotero")
def api_zotero():
    """A random sample of old papers and notes from the Zotero collection."""
    count = min(int(request.args.get("count", ZOTERO_SAMPLE)), 20)
    try:
        papers = zotero_reminder.sample_papers(count)
    except zotero_reminder.ZoteroError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:  # a Zotero outage must not kill the reading list
        logging.exception("Zotero sampling failed.")
        return jsonify({"error": str(exc)}), 502
    return jsonify({"papers": papers})


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    current = state()
    if current.refreshing:
        return jsonify({"status": "already refreshing"})
    threading.Thread(target=current.refresh, daemon=True).start()
    return jsonify({"status": "started"})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--model", type=str, default="scorer",
        help="Path to the transformer scoring model, 'none' to disable.")
    parser.add_argument(
        "--tfidf", type=str, default="vectorizer.pkl",
        help="Path to the TF-IDF vectorizer, 'none' to disable.")
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Abstracts scored at once; lower it if the scoring runs out of "
             "memory.")
    parser.add_argument(
        "--chunk-size", type=int, default=64,
        help="Scores are written to the database after every chunk.")
    parser.add_argument(
        "--threshold", type=float, default=0.01,
        help="Papers scored below this never reach the reading list.")
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Retrieve papers newer than this instead of the stored date.")
    parser.add_argument("--db", type=str, default=digest_core.DB_PATH)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="Serve the current reading list without contacting arXiv.")
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not open the web interface in a browser.")
    args = parser.parse_args()

    connection = digest_core.connect(args.db)
    scorer = None
    if args.model.lower() != "none":
        scorer = digest_core.Scorer(args.model, args.batch_size)
    highlighter = digest_core.Highlighter(
        None if args.tfidf.lower() == "none" else args.tfidf)
    app.config["state"] = State(connection, args.db, highlighter, scorer,
                                args.threshold, args.chunk_size)

    if args.no_fetch:
        # Abstracts left unscored by an interrupted run still get scored.
        if scorer is not None:
            digest_core.score_new(
                connection, scorer, args.threshold, args.chunk_size)
        expired = digest_core.expire_old(connection)
        logging.info("Skipping arXiv, %d papers expired.", expired)
    else:
        digest_core.fetch_and_score(
            connection, scorer, args.threshold, args.start_date,
            args.chunk_size)

    pending = digest_core.counts(connection).get("pending", 0)
    logging.info("Reading list has %d papers.", pending)

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    logging.info("Serving the reading list at %s", url)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
