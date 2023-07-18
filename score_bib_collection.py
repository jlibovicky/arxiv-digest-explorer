#!/usr/bin/env python3

import argparse
import logging

import bibtexparser
from pylatexenc.latex2text import LatexNodes2Text
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('anthology_bib')
    parser.add_argument('model')
    args = parser.parse_args()

    logging.info("Load the scoring model from '%s'.", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()
    logging.info("Model loaded.")

    logging.info("Load and parse the bib file from '%s'.", args.anthology_bib)
    library = bibtexparser.load(open(args.anthology_bib))

    latex2text = LatexNodes2Text()

    logging.info("Score the bib entries.")
    scored = []
    for entry in tqdm(library.entries):
        if 'abstract' in entry:
            entry['abstract'] = entry['abstract'].replace('\n', ' ')
        else:
            entry['abstract'] = ''
        title = latex2text.latex_to_text(entry['title'])
        abstract = latex2text.latex_to_text(entry['abstract'])
        prompt = f"Title: {title}. Abstract: {abstract}"
        tokenized = tokenizer(prompt, return_tensors="pt", truncation=True)
        score = F.softmax(model(**tokenized)[0], dim=1)[0, 1]
        scored.append((score.item(), title))

    logging.info("Scoring done. Sort the bib entries by score.")
    scored.sort(reverse=True)
    for score, title in scored:
        print(f"{score:.4f}   {title}")

    logging.info("Done.")


if __name__ == '__main__':
    main()
