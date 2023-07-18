#!/usr/env/bin python3

import argparse
import json
import logging
import pickle

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def load_abstracts(f_handle, exclude=None):
    arxiv_ids = set()
    abstracts = []
    for line in f_handle:
        item = json.loads(line.strip())
        # Remove version number from arxiv id.
        arxiv_id = item["arxiv_id"][:-2]
        if exclude and arxiv_id in exclude:
            continue
        if arxiv_id in arxiv_ids:
            continue
        arxiv_ids.add(arxiv_id)
        abstract_text = item["abstract"].replace("\n", " ").lower()
        abstracts.append(f"{item['title'].lower()} {abstract_text}.")
    return abstracts, arxiv_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("positive", type=argparse.FileType('r'))
    parser.add_argument("negative", type=argparse.FileType('r'))
    parser.add_argument("--stopwords", type=argparse.FileType('r'), default=None)
    args = parser.parse_args()

    stop_words = list(text.ENGLISH_STOP_WORDS)
    if args.stopwords is not None:
        stop_words = list(text.ENGLISH_STOP_WORDS.union(args.stopwords.read().splitlines()))

    logging.info("Loading abstracts.")
    positive_abstracts, positive_ids = load_abstracts(args.positive)
    negative_abstracts, _ = load_abstracts(args.negative, exclude=positive_ids)

    logging.info("Positive abstracts: %d", len(positive_abstracts))
    logging.info("Negative abstracts: %d", len(negative_abstracts))

    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000)
    logging.info("Train TF-IDF.")
    vectorizer.fit(positive_abstracts + negative_abstracts)

    vocab = vectorizer.get_feature_names_out()
    logging.info("Vocab size: %d", len(vocab))
    #for word in vocab:
    #    print(word)

    logging.info("Save the vectorizer.")
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    logging.info("Done.")

if __name__ == "__main__":
    main()
