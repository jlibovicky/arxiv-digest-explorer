#!/usr/bin/env python3


import argparse
from dateutil.parser import parse as date_parse
import json
import logging
import os
import pickle
import re
import textwrap
import time
import webbrowser

import arxiv


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


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
    retrieve_count = 500
    found_recent = None

    client = arxiv.Client(
        page_size=500,
        num_retries=5,
        delay_seconds=5.0,
    )

    while found_recent is None:
        search = arxiv.Search(
          query = "cs.CL",
          max_results = retrieve_count,
          sort_by = arxiv.SortCriterion.LastUpdatedDate
        )

        found_recent = []
        for result in client.results(search):
            if result.updated <= start_date:
                break
            found_recent.append(result)

        if retrieve_count >= 10000:
            logging.info("Reached maximum arXiv retrieval limit.")
            break
        if len(found_recent) == retrieve_count:
            logging.info("There is more recent pre-prints than retrieved, wait 3s.")
            time.sleep(3)
            retrieve_count += 500
            found_recent = None

    assert found_recent is not None

    output = []
    for result in found_recent:
        output.append({
            "title": result.title,
            "authors": ", ".join(aut.name for aut in result.authors),
            "arxiv_id": result.entry_id.split("/")[-1],
            "abstract": result.summary,
            "date": result.updated.isoformat(),
            "url": result.entry_id.replace("http", "https"),
            "comment": result.comment
        })

    output.reverse()
    return output


def print_wrapped(string):
    for line in textwrap.wrap(
            string, width=80,
            initial_indent="  ", subsequent_indent="  "):
        print(line)


def highlight_keywords(text, keep_bold_yellow=False):
    for regex in RE_KEYWORDS:
        if regex.search(text):
            text = regex.sub(
                color.RED + "\\g<0>" +
                (color.YELLOW if keep_bold_yellow else color.END), text)
    return text


def highlight_tf_idf(text, vectorizer, top_n=10):
    # Get top n tf-idf words
    tfidf = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    topn_ids = tfidf.indices[tfidf.data.argsort()[-top_n:]]
    topn_words = [feature_names[i] for i in topn_ids]

    # Highlight them with yellow
    for word in topn_words:
        text = re.sub(r"\b" + word + r"\b", color.DARKCYAN + word + color.END, text)
    return text


def score_and_filter(items, model, tokenizer, threshold):
    import torch.nn.functional as F
    filtered = []
    for item in items:
        abstract_text = item["abstract"].replace("\n", " ")
        prompt = f"Title: {item['title']}. Abstract: {abstract_text}."
        tokenized = tokenizer(prompt, return_tensors="pt", truncation=True)
        score = F.softmax(model(**tokenized)[0], dim=1)[0, 1]
        item["score"] = score.item()
        if score.item() >= threshold:
            filtered.append(item)
    return filtered


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument(
        "--tfidf", default=None, type=str,
        help="Path to the TF-IDF model vectorizer.")
    args = parser.parse_args()

    model = None
    tokenizer = None
    if args.model is not None:
        logging.info("Import Transformers and Torch.")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch.nn.functional as F
        logging.info("Load the scoring model.")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(args.model)
        model.eval()
        logging.info("Loaded.")

    vectorizer = None
    if args.tfidf is not None:
        logging.info("Import scikit-learn.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        logging.info("Load the TF-IDF vectorizer.")
        with open(args.tfidf, "rb") as f_tfidf:
            vectorizer = pickle.load(f_tfidf)
            logging.info("Loaded.")

    if args.start_date is None:
        with open("last_date") as f_date:
            start_date = date_parse(f_date.read())
    else:
        start_date = date_parse(args.start_date)
    logging.info("The youngest checked paper was %s.", start_date)
    logging.info("Retrieving abstracts from arXiv.")
    items = retrieve_recent(start_date)
    logging.info("Downloading finished, retrieved %d abstracts.", len(items))

    if args.model is not None:
        logging.info("Score the abstracts.")
        items = score_and_filter(items, model, tokenizer, args.threshold)
        logging.info("Scoring finished, %d abstracts left.", len(items))
    logging.info("Present the abstracts.")

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
        print_wrapped(
            highlight_keywords(color.YELLOW + color.BOLD + item['title'],
                               keep_bold_yellow=True) + color.END)
        print()
        print(color.BOLD + "Authors:" + color.END)
        print_wrapped(item['authors'])
        print()
        if "comment" in item and item["comment"]:
            print(color.BOLD + "Comment:" + color.END)
            print_wrapped(item['comment'])
            print()
        if "score" in item:
            score_start_color = ""
            score_end_color = ""
            print(color.BOLD + "Score: " + color.END + f"{100 * item['score']:.0f}%")
            print()

        print(color.BOLD + "Abstract:" + color.END)
        abstract = item['abstract']
        if vectorizer is not None:
            abstract = highlight_tf_idf(abstract, vectorizer)
        print_wrapped(highlight_keywords(abstract))
        print()
        answer = input("Do you want to read this? y/n? ")
        while answer not in ["y", "n"]:
            answer = input("Say 'y' or 'n'. ")

        if answer == "y":
            webbrowser.open(item['url'], new=2, autoraise=False)
            with open("positive.jsonl", "a") as f_positive:
                print(json.dumps(item), file=f_positive)
        elif answer == "n":
            with open("negative.jsonl", "a") as f_negative:
                print(json.dumps(item), file=f_negative)

        with open("last_date", "w") as f_last:
            print(item["date"], file=f_last)


if __name__ == "__main__":
    main()
