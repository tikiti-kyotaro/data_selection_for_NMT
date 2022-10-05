import argparse
import random
from typing import Dict

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="xmlr", help="model name")
parser.add_argument("--model_path", type=str, default=None, help="path to trained model")
parser.add_argument("--task", type=str, default=None, help="task name")
parser.add_argument("--format", type=str, default="csv", help="input file format")
parser.add_argument("--train", type=str, help="path to training data")
parser.add_argument("--valid", type=str, help="path to dev data")
parser.add_argument("--test", type=str, help="path to test data")
parser.add_argument("--num_labels", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epoch", type=int, default=100)
parser.add_argument("--warmup", type=float, default=0.1)
parser.add_argument("--learning_rate", type=float, default=1e-05)
parser.add_argument("--sampling_rate", type=float, default=1.0)
parser.add_argument("--early_stopping", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()


TASK_NAME = args.task

tokenizer = AutoTokenizer.from_pretrained(args.model)
if args.model_path:
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
else:
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)


def torch_fix_seed(seed: int = 42) -> None:
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def _load_dataset(
    train_path: str,
    valid_path: str,
    test_path: str,
    file_format: str = "csv",
) -> Dict:
    dataset = load_dataset(
        file_format,
        data_files={
            "train": train_path,
            "valid": valid_path,
            "test": test_path,
        },
    )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    return tokenized_datasets


def tokenize_function(examples):
    if TASK_NAME == "XNLI":
        inputs = [(premise, hypo) for premise, hypo in zip(examples["premise"], examples["hypo"])]
    else:
        inputs = examples["sentence"]
    return tokenizer(inputs, padding="max_length", truncation=True, max_length=512)


def compute_metrics(eval_pred: EvalPrediction) -> Dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def train():
    torch_fix_seed(seed=args.seed)

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        learning_rate=args.learning_rate,  # 5e-05
        num_train_epochs=args.max_epoch,
        warmup_ratio=args.warmup,
    )

    dataset = _load_dataset(
        file_format=args.format,
        train_path=args.train,
        valid_path=args.valid,
        test_path=args.test,
    )

    optim = AdamW(model.parameters(), lr=args.learning_rate)
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping)],
        optimizers=(optim, None),
    )

    # training
    trainer.train()
    metrics = trainer.evaluate(dataset["valid"])
    print("Best validation scores")
    print(metrics)

    metrics = trainer.evaluate(dataset["test"])
    print("Best test scores")
    print(metrics)


if __name__ == "__main__":
    train()
