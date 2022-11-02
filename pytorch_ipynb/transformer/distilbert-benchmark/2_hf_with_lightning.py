import lightning as L
import os
import torch
import time
import torchmetrics
from torch.utils.data import DataLoader, Dataset

# from local shared.py
from shared import download_data, prepare_data

# HF libraries
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset


def tokenize_text(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)


class IMDBDataset(Dataset):
    def __init__(self, dataset_dict, partition_key="train"):
        self.partition = dataset_dict[partition_key]

    def __getitem__(self, index):
        return self.partition[index]

    def __len__(self):
        return self.partition.num_rows


if __name__ == "__main__":
    download_data()
    prepare_data()


    imdb_dataset = load_dataset(
        "csv",
        data_files={
            "train": "train.csv",
            "validation": "validation.csv",
            "test": "test.csv",
        },
    )

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    imdb_tokenized = imdb_dataset.map(tokenize_text, batched=True, batch_size=None)
    del imdb_dataset

    ##########################
    ## NEW: Dataloaders
    ##########################

    imdb_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_dataset = IMDBDataset(imdb_tokenized, partition_key="train")
    val_dataset = IMDBDataset(imdb_tokenized, partition_key="validation")
    test_dataset = IMDBDataset(imdb_tokenized, partition_key="test")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=12,
        shuffle=True, 
        num_workers=4)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=12,
        num_workers=4
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=12,
        num_workers=4
    )

    ###############################
    ## NEW: Lightning Model
    ###############################

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2)

    class LightningModel(L.LightningModule):
        def __init__(self, model, learning_rate=5e-5):
            super().__init__()

            self.learning_rate = learning_rate
            self.model = model

            self.val_acc = torchmetrics.Accuracy()
            self.test_acc = torchmetrics.Accuracy()

        def forward(self, input_ids, attention_mask, labels):
            return self.model(input_ids, attention_mask=attention_mask, labels=labels)

        def training_step(self, batch, batch_idx):
            outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                           labels=batch["label"])        
            self.log("train_loss", outputs["loss"])
            return outputs["loss"]  # this is passed to the optimizer for training

        def validation_step(self, batch, batch_idx):
            outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                           labels=batch["label"])        
            self.log("val_loss", outputs["loss"], prog_bar=True)

            logits = outputs["logits"]
            predicted_labels = torch.argmax(logits, 1)
            self.val_acc(predicted_labels, batch["label"])
            self.log("val_acc", self.val_acc, prog_bar=True)

        def test_step(self, batch, batch_idx):
            outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                           labels=batch["label"])        

            logits = outputs["logits"]
            predicted_labels = torch.argmax(logits, 1)
            self.test_acc(predicted_labels, batch["label"])
            self.log("accuracy", self.test_acc, prog_bar=True)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return optimizer

    lightning_model = LightningModel(model)

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices="auto",
        strategy="ddp"
    )

    start = time.time()
    trainer.fit(model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    train_time = (time.time()-start)/60

    start = time.time()
    outputs = trainer.test(lightning_model, dataloaders=test_loader)
    inf_time = (time.time()-start)/60
    print(outputs)

    print("======================")
    print(f"Training time: {train_time:.2f}")
    print(f"Inference time: {inf_time:.2f}")

    print("======================")
    print("Lightning", L.__version__)
