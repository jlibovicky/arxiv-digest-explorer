#!/usr/bin/env python3

import argparse
import logging
import os
import random
import sys
import webbrowser

from pyzotero import zotero


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


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


def format_authors(authors):
    """Format author names for display"""
    author_list = []
    for author in authors:
        if author['creatorType'] != 'author':
            continue
        author_list.append(f"{author['firstName']} {author['lastName']}")
    return ", ".join(author_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument("--sample-papers", type=int, default=3)
    parser.add_argument(
        "--no-open", action="store_true", default=False,
        help="Do not open the output file in a browser.")
    args = parser.parse_args()

    logging.info("Loading Zotero API key and library ID.")
    if not os.path.exists("zotero_api_key.txt"):
        logging.error("zotero_api_key.txt not found.")
        return
    with open("zotero_api_key.txt", "r") as f:
        api_key = f.read().strip()
    if not os.path.exists("zotero_library_id.txt"):
        logging.error("zotero_library_id.txt not found.")
        return
    with open("zotero_library_id.txt", "r") as f:
        library_id = f.read().strip()
    if not os.path.exists("zotero_collection_id.txt"):
        logging.error("zotero_collection_id.txt not found.")
        return
    with open("zotero_collection_id.txt", "r") as f:
        collection_id = f.read().strip()

    logging.info("Connecting to Zotero API.")
    zot = zotero.Zotero(library_id, 'user', api_key)

    logging.info("Fetching items from the Zotero library.")
    items = zot.everything(zot.collection_items(
        collection_id, itemType="conferencePaper || journalArticle || report || preprint"))
    logging.info(f"Found {len(items)} items, sampling {args.sample_papers}.")

    items = random.sample(items, args.sample_papers)

    logging.info("Generating HTML.")
    print(HEADER, file=args.output)

    for item in items:
        url = item['data']['url']
        title = item['data']['title']
        print(f"<h1><a href='{url}'>{title}</a></h1>", file=args.output)
        print(f"<p><b>{format_authors(item['data']['creators'])}</b></p>", file=args.output)

        print("<p>", file=args.output)
        for tag in item['data']['tags']:
            print(f"<span class='badge badge-primary'>{tag['tag']}</span>", file=args.output)
        print("</p>", file=args.output)

        if 'abstractNote' in item['data'] and item['data']['abstractNote']:
            print('<div class="abstract">', file=args.output)
            print(f"<b>Abstract:</b> {item['data']['abstractNote']}</div>", file=args.output)

        note = zot.children(item['key'], itemType="note")
        if note:
            print('<div class="note">', file=args.output)
            print(f"<b>My notes:</b> {note[0]['data']['note']}", file=args.output)
            print("</div>", file=args.output)
        print("<hr />", file=args.output)

    print(FOOTER, file=args.output)
    args.output.close()

    if args.output != sys.stdout and not args.no_open:
        filename = args.output.name
        webbrowser.open(filename, new=2, autoraise=False)

    logging.info("Done.")

if __name__ == "__main__":
    main()
