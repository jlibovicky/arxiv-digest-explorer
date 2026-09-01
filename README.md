# arxiv-digest-explorer

Tool for triaging new cs.CL pre-prints that collects data for future
automation.

## Web interface (`webapp.py`)

```
./webapp.py                       # scorer/ + vectorizer.pkl by default
./webapp.py --model none --tfidf none --no-fetch
```

On start the app downloads pre-prints newer than the last one it has seen,
stores them in `reading_list.db`, scores them with the transformer
classifier and puts everything above `--threshold` on the reading list. It
then opens a browser with the list ordered by the predicted score, best paper
first.

Give `--start-date` only for the very first run: it overrides the stored
date and downloads everything since then again (already known abstracts are
recognised and not scored again, but they are transferred again). Without it
the app continues where the last run stopped, which costs a single page of
200 results.

Downloading and scoring are separate steps. Abstracts are stored and the
"newest seen" date is moved as soon as the download finishes, and scores are
written to the database after every chunk of `--chunk-size` abstracts, so an
interrupted or crashed run never downloads or scores the same abstract twice
-- just start the app again and it continues where it stopped. Papers below
the threshold stay in the database as `rejected` so that they are not
downloaded again either.

If the scoring runs out of memory, lower `--batch-size` (8 by default, needs
about 1 GB of RAM).

For each paper:

* `y` — open the pre-print in a new tab and log it as positive,
* `n` — log it as negative,
* `l` or space — remind me later: nothing is logged and the paper shows up
  again the next time the app starts.

* `z` — take a break and read a few random old papers and notes from the
  Zotero collection (`Esc` or a click outside closes the panel, "Another
  sample" draws new ones); the same sampling is available from the
  `Zotero reminder` button in the header.

Papers left untouched for four months quietly disappear from the reading list
without being logged either way.

Keywords from `digest_core.KEYWORDS` and the top TF-IDF terms of the abstract
are highlighted, same as in the CLI version. The decisions are appended to
`positive.jsonl` and `negative.jsonl`, which is what the `train_*.py` scripts
consume.

Useful options: `--threshold`, `--batch-size`, `--chunk-size`,
`--start-date` (needed for the very first run if there is no `last_date`
file; `01.08.2026` and `2026-08-01` both mean the first of August), `--db`,
`--port`, `--no-fetch` (serve the list without contacting arXiv; still scores
whatever an earlier run left unscored), `--no-browser`.

## Zotero reminder (`zotero_reminder.py`)

Samples random papers with your notes from a Zotero collection, given
`zotero_api_key.txt`, `zotero_library_id.txt` and `zotero_collection_id.txt`
in the working directory. Run it on its own to get a standalone HTML page:

```
./zotero_reminder.py --sample-papers 3 reminder.html
```

The web app calls the same `sample_papers` function from `/api/zotero`, so
the reminders work from the browser as well (key `z`). If `pyzotero` or the
credential files are missing, only that panel reports the problem; the
reading list keeps working.

## CLI (`explore.py`)

The original chronological version: it walks through the abstracts in the
terminal and asks `y`/`n` for each of them. It shares nothing with the web app
except the logs.
