import time

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
from torch.optim.lr_scheduler import ExponentialLR, StepLR, MultiStepLR
import wandb

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = 'snunlp/KR-ELECTRA-discriminator'
        self.lr = 1e-5

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name, num_labels=1, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
        # Loss 계산을 위해 사용될 Loss를 호출합니다.
        self.loss_func = torch.nn.MSELoss()# L1Loss()
        
    def resize_token_embeddings(self, len_tokenizer):
        self.plm.resize_token_embeddings(len_tokenizer)
        

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9,0.999), eps=1e-6)
        return [optimizer] 