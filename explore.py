#!/usr/bin/env python3


import argparse
from dateutil.parser import parse as date_parse
import json
import os
import re
import textwrap
import webbrowser

import arxiv


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


KEYWORDS = [
    "translation", "multilingual", "cross-lingual", "multimodal",
    "tokenization", "language-vision", "vision-language"]


RE_KEYWORDS = [
    re.compile(kwrd, re.IGNORECASE)
    for kwrd in KEYWORDS]


def retrieve_recent(start_date):
    search = arxiv.Search(
      query = "cs.CL",
      max_results = 40,
      sort_by = arxiv.SortCriterion.LastUpdatedDate
    )

    output = []
    for result in search.results():
        if result.updated <= start_date:
            break

        output.append({
            "title": result.title,
            "authors": ", ".join(aut.name for aut in result.authors),
            "arxiv_id": result.entry_id.split("/")[-1],
            "abstract": result.summary,
            "date": result.updated.isoformat(),
            "url": result.entry_id.replace("http", "https")
        })

    output.reverse()
    return output


def parse(file_handle):
    items = []
    current_id = None
    current_date = None
    current_title = None
    current_authors = None
    current_abstract = None
    after_categories = False
    current_url = None
    in_abstract = False

    for _ in range(13):
        file_handle.readline()

    for line in file_handle:
        line = line.rstrip()
        if current_title is None:
            if (line == r"\\" or
                not line):
                continue
            elif line.startswith("Date:"):
                current_date = date_parse(
                    re.sub("GMT.*", "GMT", line[11:])).isoformat()
            elif line.startswith("arXiv:"):
                current_id = line[6:16]
            elif line.startswith("Title: "):
                current_title = line[7:]
        elif current_authors is None:
            if line.startswith("Authors: "):
                current_authors = line[9:]
            else:
                current_title += " " + line.lstrip()
        elif not after_categories:
            if line.startswith("Categories:"):
                after_categories = True
            else:
                current_authors += ", " + line.lstrip()
        elif not in_abstract and current_abstract is None:
            if line == r"\\":
                in_abstract = True
        elif in_abstract:
            if current_abstract is None:
                current_abstract = line.lstrip()
            elif line.startswith(r"\\"):
                current_url = line[5:37]
                in_abstract = False
            else:
                current_abstract += " " + line
        elif (line.startswith("------------------") or
              line.startswith("%-%-%-%-%-%-%-%-%") or
              line.startswith("%%--%%--%%--%%--%%--%%--")):
            items.append({
                "arxiv_id": current_id,
                "date": current_date,
                "title": current_title,
                "authors": current_authors,
                "url": current_url,
                "abstract": current_abstract
            })
            if line.startswith("%%--%%--%%--%%--%%--%%--"):
                break
            current_id = None
            current_date = None
            current_title = None
            current_authors = None
            current_abstract = None
            after_categories = False
            current_url = None
            in_abstract = False
        else:
            raise RuntimeError("This should no happen.")

    return items


def print_wrapped(string):
    for line in textwrap.wrap(
            string, width=80,
            initial_indent="  ", subsequent_indent="  "):
        print(line)


def highlight_keywords(text):
    for regex in RE_KEYWORDS:
        if regex.search(text):
            text = regex.sub(color.RED + "\\g<0>" + color.END, text)
    return text


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--start-date", type=str, default=None)
    args = parser.parse_args()

    if args.start_date is None:
        with open("last_date") as f_date:
            start_date = date_parse(f_date.read())
    else:
        start_date = date_parse(args.start_date)
    items = retrieve_recent(start_date)

    positive = []
    negative = []

    for i, item in enumerate(items):
        os.system('clear')
        print(f"{i + 1} / {len(items)}")
        print()
        print(color.BOLD + "arXiv ID: " + color.END + item['arxiv_id'])
        print(
            color.BOLD + "Date: " + color.END +
            date_parse(item['date']).strftime("%d.%m.%Y, %H:%M:%S"))
        print()
        print(color.BOLD + "Title:" + color.END)
        print_wrapped(highlight_keywords(item['title']))
        print()
        print(color.BOLD + "Authors:" + color.END)
        print_wrapped(item['authors'])
        print()
        print(color.BOLD + "Abstract:" + color.END)
        print_wrapped(highlight_keywords(item['abstract']))
        print()
        answer = input("Do you want to read this? y/n? ")
        while answer not in ["y", "n"]:
            answer = input("Say 'y' or 'n'. ")

        if answer == "y":
            positive.append(item)
            webbrowser.open(item['url'], new=2, autoraise=False)
        elif answer == "n":
            negative.append(item)

        with open("last_date", "w") as f_last:
            print(item["date"], file=f_last)

    with open("positive.jsonl", "a") as f_positive:
        for item in positive:
            print(json.dumps(item), file=f_positive)
    with open("negative.jsonl", "a") as f_negative:
        for item in negative:
            print(json.dumps(item), file=f_negative)


if __name__ == "__main__":
    main()
