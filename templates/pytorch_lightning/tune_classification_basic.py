import time
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import torch
from torch.utils.data import DataLoader
import torchmetrics
from torchvision import transforms
from torchvision import datasets
from torch.utils.data.dataset import random_split


# Argparse helper
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
parser.add_argument("--epochs", type=int, default=10, help="Num. of epochs.")
parser.add_argument("--workers", type=int, default=4, help="Num. of workers.")
parser.add_argument("--learning_rate", type=int, default=0.005, help="Learning rate.")
args = parser.parse_args()
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
NUM_WORKERS = args.workers


##################################################################
# PYTORCH MODULE
##################################################################

# Regular PyTorch Module
class PyTorchModel(torch.nn.Module):
    def __init__(self, input_size, hidden_units, num_classes):
        super().__init__()

        # Initialize MLP layers
        all_layers = []
        for hidden_unit in hidden_units:
            layer = torch.nn.Linear(input_size, hidden_unit, bias=False)
            all_layers.append(layer)
            all_layers.append(torch.nn.ReLU())
            input_size = hidden_unit

        output_layer = torch.nn.Linear(
            in_features=hidden_units[-1], out_features=num_classes
        )

        all_layers.append(output_layer)
        self.layers = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # to make it work for image inputs
        x = self.layers(x)
        return x  # x are the model's logits


##################################################################
# PYTORCH LIGHTNING MODULE
##################################################################

# LightningModule that receives a PyTorch model as input
class LightningModel(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        # The inherited PyTorch module
        self.model = model
        if hasattr(model, "dropout_proba"):
            self.dropout_proba = model.dropout_proba

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=["model"])

        # Set up attributes for computing the accuracy
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    # Defining the forward method is only necessary
    # if you want to use a Trainer's .predict() method (optional)
    def forward(self, x):
        return self.model(x)

    # A common forward step to compute the loss and labels
    # this is used for training, validation, and testing below
    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        loss = torch.nn.functional.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)

        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)

        # Do another forward pass in .eval() mode to compute accuracy
        # while accountingfor Dropout, BatchNorm etc. behavior
        # during evaluation (inference)
        self.model.eval()
        with torch.no_grad():
            _, true_labels, predicted_labels = self._shared_step(batch)
        self.train_acc(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        self.model.train()

        return loss  # this is passed to the optimzer for training

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("valid_loss", loss)
        self.valid_acc(predicted_labels, true_labels)
        self.log(
            "valid_acc",
            self.valid_acc,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


##################################################################
# DATA MODULE
##################################################################


class DataModule(pl.LightningDataModule):
    def __init__(self, data_path="./"):
        super().__init__()
        self.data_path = data_path

    def prepare_data(self):
        datasets.MNIST(root=self.data_path, download=True)
        return

    def setup(self, stage=None):
        # Note transforms.ToTensor() scales input images
        # to 0-1 range
        train = datasets.MNIST(
            root=self.data_path,
            train=True,
            transform=transforms.ToTensor(),
            download=False,
        )

        self.test = datasets.MNIST(
            root=self.data_path,
            train=False,
            transform=transforms.ToTensor(),
            download=False,
        )

        self.train, self.valid = random_split(train, lengths=[55000, 5000])

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train,
            batch_size=BATCH_SIZE,
            drop_last=True,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            dataset=self.valid,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        return test_loader


##################################################################
# INITIALIZE MODELS AND TRAINER
##################################################################

pytorch_model = PyTorchModel(input_size=28 * 28, hidden_units=(128, 256), num_classes=10)

lightning_model = LightningModel(pytorch_model, learning_rate=LEARNING_RATE)

callbacks = [
    ModelCheckpoint(save_top_k=1, mode="max", monitor="valid_acc")
]  # save top 1 model
logger = CSVLogger(save_dir="logs/", name="my-model")

torch.manual_seed(1)
data_module = DataModule(data_path="./data")

trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    callbacks=callbacks,
    progress_bar_refresh_rate=50,  # recommended for notebooks
    accelerator="auto",  # Uses GPUs or TPUs if available
    devices="auto",  # Uses all available GPUs/TPUs if applicable
    logger=logger,
    deterministic=True,
    log_every_n_steps=10,
)

##################################################################
# TRAIN AND EVALUATE
##################################################################

start_time = time.time()
trainer.fit(model=lightning_model, datamodule=data_module)

runtime = (time.time() - start_time) / 60
print(f"Training took {runtime:.2f} min in total.")

trainer.test(model=lightning_model, datamodule=data_module, ckpt_path="best")
