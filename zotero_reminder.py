#!/usr/bin/env python3
"""Show a random sample of old papers and notes from a Zotero collection."""

import argparse
import html
import logging
import os
import random
import sys
import webbrowser


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


ITEM_TYPES = "conferencePaper || journalArticle || report || preprint"

CREDENTIAL_FILES = {
    "api_key": "zotero_api_key.txt",
    "library_id": "zotero_library_id.txt",
    "collection_id": "zotero_collection_id.txt",
}


# Header that imports Bootstrap CSS
HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<title>Zotero Snapshot</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
<style>
.abstract {
    font-size: 0.8em;
    margin-bottom: 1em;
    background-color: #f0f0f0;
}
.note {
}
</style>
</head>
<body style="padding: 20px;">
"""

FOOTER = """
</body>
</html>
"""


class ZoteroError(RuntimeError):
    """Something is missing or the Zotero API did not cooperate."""


def format_authors(authors):
    """Format author names for display"""
    author_list = []
    for author in authors:
        if author['creatorType'] != 'author':
            continue
        author_list.append(f"{author['firstName']} {author['lastName']}")
    return ", ".join(author_list)


def load_credentials(directory="."):
    """Read the API key, library and collection ids from the text files."""
    credentials = {}
    for name, filename in CREDENTIAL_FILES.items():
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            raise ZoteroError(f"{filename} not found.")
        with open(path, "r") as f_credential:
            credentials[name] = f_credential.read().strip()
    return credentials


def connect(directory="."):
    """Open a Zotero client and tell which collection to sample from."""
    try:
        from pyzotero import zotero
    except ImportError as exc:  # pyzotero is optional for the rest of the app
        raise ZoteroError(
            "pyzotero is not installed, run 'pip install pyzotero'.") from exc

    credentials = load_credentials(directory)
    zot = zotero.Zotero(
        credentials["library_id"], 'user', credentials["api_key"])
    return zot, credentials["collection_id"]


def sample_papers(count=3, directory="."):
    """Return `count` random papers of the collection with their notes."""
    zot, collection_id = connect(directory)

    logging.info("Fetching items from the Zotero library.")
    items = zot.everything(
        zot.collection_items(collection_id, itemType=ITEM_TYPES))
    logging.info("Found %d items, sampling %d.", len(items), count)
    if not items:
        return []

    papers = []
    for item in random.sample(items, min(count, len(items))):
        data = item['data']
        notes = zot.children(item['key'], itemType="note")
        papers.append({
            "key": item['key'],
            "title": data.get('title', ''),
            "url": data.get('url', ''),
            "authors": format_authors(data.get('creators', [])),
            "tags": [tag['tag'] for tag in data.get('tags', [])],
            "abstract": data.get('abstractNote', ''),
            # Zotero notes are HTML written by the user themselves.
            "note_html": notes[0]['data']['note'] if notes else "",
        })
    return papers


def papers_to_html(papers, output):
    """Write the standalone Bootstrap page the CLI opens in a browser."""
    print(HEADER, file=output)
    for paper in papers:
        title = html.escape(paper["title"])
        url = html.escape(paper["url"], quote=True)
        print(f"<h1><a href='{url}'>{title}</a></h1>", file=output)
        print(f"<p><b>{html.escape(paper['authors'])}</b></p>", file=output)

        print("<p>", file=output)
        for tag in paper["tags"]:
            print(f"<span class='badge badge-primary'>{html.escape(tag)}</span>",
                  file=output)
        print("</p>", file=output)

        if paper["abstract"]:
            print('<div class="abstract">', file=output)
            print(f"<b>Abstract:</b> {html.escape(paper['abstract'])}</div>",
                  file=output)

        if paper["note_html"]:
            print('<div class="note">', file=output)
            print(f"<b>My notes:</b> {paper['note_html']}", file=output)
            print("</div>", file=output)
        print("<hr />", file=output)
    print(FOOTER, file=output)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument("--sample-papers", type=int, default=3)
    parser.add_argument(
        "--no-open", action="store_true", default=False,
        help="Do not open the output file in a browser.")
    args = parser.parse_args()

    try:
        papers = sample_papers(args.sample_papers)
    except ZoteroError as exc:
        logging.error("%s", exc)
        return 1

    logging.info("Generating HTML.")
    papers_to_html(papers, args.output)
    args.output.close()

    if args.output != sys.stdout and not args.no_open:
        webbrowser.open(args.output.name, new=2, autoraise=False)

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
