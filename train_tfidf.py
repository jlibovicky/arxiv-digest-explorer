#!/usr/bin/evn python3

import argparse
import logging
import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("positive", type=argparse.FileType('r'))
    parser.add_argument("negative", type=argparse.FileType('r'))
    args = parser.parse_args()

    logging.info("Load positive instances from '%s'.", args.positive)
    positive_ids = set()
    abstracts = []
    labels = []
    for line in args.positive:
        item = json.loads(line.strip())
        positive_ids.add(item["arxiv_id"])
        abstracts.append(item["title"] + ". " +item["abstract"])
        labels.append(True)

    logging.info("Load negative instances from '%s'.", args.negative)
    negative_abstracts = []
    for line in args.negative:
        item = json.loads(line.strip())
        if item["arxiv_id"] not in positive_ids:
            abstracts.append(item["abstract"])
            labels.append(False)

    scores = []
    conf_matrices = []
    logging.info("Train model on 500 random splits.")
    for i in range(500):
        x_train, x_test, y_train, y_test = train_test_split(abstracts, labels)

        vectorizer = TfidfVectorizer(stop_words='english')
        vectors_train = vectorizer.fit_transform(x_train)
        vectors_test = vectorizer.transform(x_test)

        model = LogisticRegression(class_weight='balanced')
        model.fit(vectors_train, y_train)

        y_pred = model.predict(vectors_test)
        score = f1_score(y_test, y_pred)
        scores.append(score)

        conf_matrices.append(confusion_matrix(y_test, y_pred))

    logging.info("F1 score: %.3f +-/ %.3f", np.mean(scores), np.std(scores))
    logging.info("Confusions matrix\n%s", np.mean(conf_matrices, axis=0))
    logging.info("Done.")


if __name__ == "__main__":
    main()
