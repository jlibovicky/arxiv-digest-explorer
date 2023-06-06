#!/usr/bin/env python3

import argparse
import json
import logging
import re

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback)
from transformers import DataCollatorWithPadding


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
        # Remove version number from arxiv id.
        arxiv_id = item["arxiv_id"][:-2]
        if exclude and arxiv_id in exclude:
            continue
        if arxiv_id in arxiv_ids:
            continue
        arxiv_ids.add(arxiv_id)
        abstract_text = item["abstract"].replace("\n", " ")
        abstracts.append(
            f"Title: {item['title']}. Abstract: {abstract_text}.")
    return abstracts, arxiv_ids


class WeightedTrainer(Trainer):
    def __init__(self, positive_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_weight = positive_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, self.positive_weight]).to(device))
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro")

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("positive", type=argparse.FileType('r'))
    parser.add_argument("negative", type=argparse.FileType('r'))
    parser.add_argument("--positive-weight", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--model-name", type=str, default="distilroberta-base")
    args = parser.parse_args()

    logging.info("Loading abstracts.")
    positive_abstracts, positive_ids = load_abstracts(args.positive)
    negative_abstracts, _ = load_abstracts(args.negative, exclude=positive_ids)

    logging.info("Creating datasets.")
    abstracts_train, abstracts_test, labels_train, labels_test = train_test_split(
        positive_abstracts + negative_abstracts,
        [1] * len(positive_abstracts) + [0] * len(negative_abstracts),
        test_size=0.2, random_state=938)

    abstracts_train, abstracts_val, labels_train, labels_val = train_test_split(
        abstracts_train, labels_train, test_size=0.2, random_state=938)

    logging.info("Initialize tokenizer and tokenized datasets.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = ArxivDataset(abstracts_train, labels_train, tokenizer)
    val_dataset = ArxivDataset(abstracts_val, labels_val, tokenizer)
    test_dataset = ArxivDataset(abstracts_test, labels_test, tokenizer)

    logging.info("Loading a pre-trained model.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2,
        hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.25).to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir='./scorer',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=256,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        report_to="tensorboard",
        #logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        metric_for_best_model="f1",
        seed=1234,
    )

    trainer = WeightedTrainer(
        model=model,
        positive_weight=args.positive_weight,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()
    #test_res = trainer.evaluate(test_dataset)
    #print(test_res["eval_f1"])

    test_dataloader = DataLoader(
        test_dataset, collate_fn=data_collator, batch_size=64)
    test_scores = []
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            scores = F.softmax(
                model(**batch).logits, dim=1)[:,1].cpu().numpy().tolist()
            test_scores.extend(scores)

    for score, abstract in sorted(zip(test_scores, abstracts_test)):
        title = re.match(r"Title: (.*)\. Abstract:", abstract).group(1)
        print(f"{score:.3f}\t{title}")

    model.save_pretrained("./scorer")
    tokenizer.save_pretrained("./scorer")

if __name__ == "__main__":
    main()
