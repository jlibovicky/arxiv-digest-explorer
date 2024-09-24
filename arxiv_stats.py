#!/usr/bin/env python3


import arxiv
import logging


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


client = arxiv.Client(
    page_size=800,
    num_retries=5,
    delay_seconds=5.0,
)

search = arxiv.Search(
  query = "cs.CL",
  max_results = 30000,
  sort_by = arxiv.SortCriterion.LastUpdatedDate
)


date_counts = {}

for result in client.results(search):
    timestamp = f"{result.updated.year}-{result.updated.month:02d}"
    if timestamp not in date_counts:
        date_counts[timestamp] = 0
    date_counts[timestamp] += 1

for month, count in sorted(date_counts.items()):
    print(f"{month}\t{count}")
