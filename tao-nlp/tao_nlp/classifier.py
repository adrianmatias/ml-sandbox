import evaluate
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
    Trainer,
    TrainingArguments,
)


class Conf:
    dataset_name = "imdb"
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    metric = "accuracy"
    model_max_length = 512
    path_model = "model_finetuned"


def main():
    dataset_train, dataset_test, label_name_list = load_and_prepare_data()
    train_and_evaluate_model(dataset_train, dataset_test, label_name_list)
    analyze_predictions(dataset_test=dataset_test)


def preprocess(text: str) -> str:
    return text.lower()


def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(metric, eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def load_and_prepare_data():
    dataset = load_dataset(Conf.dataset_name)
    label_name_list = dataset["train"].features["label"].names
    tokenizer = AutoTokenizer.from_pretrained(Conf.model_name)
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(tokenizer, examples), batched=True
    )

    dataset_train = tokenized_datasets["train"].shuffle(seed=42).select(range(10))
    dataset_test = tokenized_datasets["test"].shuffle(seed=42).select(range(10))

    return dataset_train, dataset_test, label_name_list


def train_and_evaluate_model(dataset_train, dataset_test, label_name_list):
    model = AutoModelForSequenceClassification.from_pretrained(
        Conf.model_name, num_labels=len(label_name_list)
    )
    metric = evaluate.load(Conf.metric)

    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        metric_for_best_model="accuracy",
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        compute_metrics=lambda eval_pred: compute_metrics(metric, eval_pred),
    )

    train_output = trainer.train()
    print(train_output)

    print(trainer.evaluate())

    trainer.save_model(Conf.path_model)


def analyze_predictions(dataset_test):
    tokenizer = AutoTokenizer.from_pretrained(
        Conf.model_name, model_max_length=Conf.model_max_length
    )
    pipe = TextClassificationPipeline(
        model=AutoModelForSequenceClassification.from_pretrained(Conf.path_model),
        tokenizer=tokenizer,
        padding=True,
        truncation=True,
    )

    text_list = [rec["text"] for rec in dataset_test]
    label_list = [rec["label"] for rec in dataset_test]

    df = pd.DataFrame.from_dict(
        {
            "label": label_list,
            "pred": [pred["label"] for pred in pipe(text_list)],
            "text": dataset_test["text"],
        }
    )
    print(df)


if __name__ == "__main__":
    main()
