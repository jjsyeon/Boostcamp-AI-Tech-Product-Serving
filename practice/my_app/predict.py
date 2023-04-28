import torch
import pytorch_lightning as pl

import streamlit as st

import yaml
from typing import Tuple

import dataloader as data_loader
import model as MM


@st.cache
def load_model() -> MM.Model:
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(config['model_path'], map_location=device)
    return model

def get_prediction(model:MM.Model, sentence1:str, sentence2:str) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = data_loader.stDataloader(sentence1, sentence2)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=1, log_every_n_steps=100)
    predictions = trainer.predict(model=model, datamodule=dataloader)
    predictions = round(float(predictions[0].tolist()),1)
    return predictions

# if __name__ == '__main__':
#     model = load_model()
#     outputs = get_prediction(model,"안녕하십니까", "안녕하세요!")

#     print(outputs)