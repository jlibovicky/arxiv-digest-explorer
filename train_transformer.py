#!/usr/bin/env python3

import argparse
import json
import logging

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback)


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class ArxivDataset(torch.utils.data.Dataset):
    def __init__(self, abstracts, lables, tokenizer):
        self.abstracts = abstracts
        self.lables = lables
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        abstract = self.abstracts[index]
        label = torch.tensor(self.lables[index])
        encoding = self.tokenizer(
            abstract, padding='max_length', truncation=True)
        encoding_dict = {
            key: torch.tensor(val) for key, val in encoding.items()}
        encoding_dict["labels"] = label
        return encoding_dict

    def __len__(self):
        return len(self.abstracts)


def load_abstracts(f_handle, exclude=None):
    arxiv_ids = set()
    abstracts = []
    for line in f_handle:
        item = json.loads(line.strip())
        arxiv_id = item["arxiv_id"]
        if exclude and arxiv_id in exclude:
            continue
        arxiv_ids.add(arxiv_id)
        abstracts.append(item["title"] + ". " +item["abstract"])
    return abstracts, arxiv_ids


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    prec, rec, f1, support = precision_recall_fscore_support(
        labels, predictions, average="macro")

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": support,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("positive", type=argparse.FileType('r'))
    parser.add_argument("negative", type=argparse.FileType('r'))
    args = parser.parse_args()

    logging.info("Loading abstracts.")
    positive_abstracts, positive_ids = load_abstracts(args.positive)
    negative_abstracts, _ = load_abstracts(args.negative, exclude=positive_ids)

    logging.info("Creating datasets.")
    abstracts_train, abstracts_test, labels_train, labels_test = train_test_split(
        positive_abstracts + negative_abstracts,
        [1] * len(positive_abstracts) + [0] * len(negative_abstracts),
        test_size=0.2)

    abstracts_train, abstracts_val, labels_train, labels_val = train_test_split(
        abstracts_train, labels_train, test_size=0.2)

    logging.info("Initialize tokenizer and tokenized datasets.")
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    train_dataset = ArxivDataset(abstracts_train, labels_train, tokenizer)
    val_dataset = ArxivDataset(abstracts_val, labels_val, tokenizer)
    test_dataset = ArxivDataset(abstracts_test, labels_test, tokenizer)

    logging.info("Loading a pre-trained model.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base").to(device)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=256,
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()


if __name__ == "__main__":
    main()
