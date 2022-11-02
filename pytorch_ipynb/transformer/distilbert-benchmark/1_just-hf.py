import numpy as np
import torch
import time

# from local shared.py
from shared import download_data, prepare_data

# HF libraries
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_metric, load_dataset


def tokenize_text(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = metric.compute(predictions=predictions, references=labels)
    return {"accuracy": acc}

if __name__ == "__main__":
    download_data()
    prepare_data()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print("======================")
    print("Device", device)
    print("======================")

    imdb_dataset = load_dataset(
        "csv",
        data_files={
            "train": "train.csv",
            "validation": "validation.csv",
            "test": "test.csv",
        },
    )

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    imdb_tokenized = imdb_dataset.map(tokenize_text, batched=True, batch_size=12)
    del imdb_dataset

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2)
    model.to(device)

    metric = load_metric("accuracy")

    trainer_args = TrainingArguments(output_dir="distilbert-v1",
                                     num_train_epochs=3,
                                     evaluation_strategy="epoch",
                                     per_device_train_batch_size=12,
                                     per_device_eval_batch_size=12,
                                     learning_rate=1e-5)

    trainer = Trainer(model=model,
                      args=trainer_args,
                      compute_metrics=compute_metrics,
                      train_dataset=imdb_tokenized["train"],
                      eval_dataset=imdb_tokenized["validation"],
                      tokenizer=tokenizer)

    start = time.time()
    trainer.train()
    train_time = (time.time()-start)/60

    start = time.time()
    outputs = trainer.predict(imdb_tokenized["test"])
    inf_time = (time.time()-start)/60
    print(outputs.metrics)

    print("======================")
    print(f"Training time: {train_time:.2f}")
    print(f"Inference time: {inf_time:.2f}")

    print("======================")
    print("Transformers", transformers.__version__)