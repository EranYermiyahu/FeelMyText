from dataset import DataSet
from bert_ect import EmotionClassifier
from Trainer import Trainer
from Model import TransformerECT
import torch.nn as nn
from transformers import RobertaForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader

EPOCHS = 10
ptm_output_dim = 28
num_classes = 7

def objective(trial, device, train_loader, val_loader):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    n_units = trial.suggest_int("n_units", 4, 64)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    model = EmotionClassifier(num_classes=7,
                              dropout=dropout,
                              is_roberta=True,
                              feature_extracting=False,
                              mlp_enable=True,
                              n_layers=n_layers,
                              n_units=n_units)
    
    trainer = Trainer(model,
                      train_loader,
                      val_loader,
                      device,
                      batch_size=batch_size,
                      learning_rate=lr,
                      epochs=EPOCHS,
                      optuna=True,
                      trial=trial)
    
    train_losses_list, val_acc_list = trainer.train()

    return val_acc_list[-1]





    

    

    
    



