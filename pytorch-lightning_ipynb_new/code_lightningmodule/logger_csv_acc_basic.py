from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


lightning_model = LightningModel(pytorch_model, learning_rate=LEARNING_RATE)

callbacks = [
    ModelCheckpoint(
        save_top_k=1, mode="max", monitor="valid_acc"
    )  # save top 1 model
]
logger = CSVLogger(save_dir="logs/", name="my-model")
