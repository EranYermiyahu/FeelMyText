import torch
import pandas as pd
from dataset import DataSet
from bert_ect import EmotionClassifier
from Trainer import Trainer
from transformers import BertForSequenceClassification
from Model import TransformerECT
from torch.utils.data import TensorDataset, DataLoader

def check_gpu():
    # Check if GPU is available and being used
    print(torch.cuda.is_available())

    # Check the number of available GPUs
    print(torch.cuda.device_count())

    # Get the name of the current GPU
    print(torch.cuda.get_device_name(torch.cuda.current_device()))


BATCH_SIZE = 64
EPOCHS = 10
LR = 2e-5
DROPOUT = [0.1, 0.25]


if __name__ == '__main__':
    # check_gpu()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset = DataSet()
    dataset.preprocessing_data(generate_from_scratch=False, data_augmentation=True, force_equality=False)
    # dataset.remove_unclear_samples()
    # dataset.add_emotion_label()
    dataset.count_labels()
    dataset.tokenizer()
    train_dataset, val_dataset, test_dataset = dataset.split_train_test_val_data()
    # Create data loaders for train, test, and validation sets
    train_loader, test_loader, val_loader = dataset.create_data_loaders(train_dataset, val_dataset, test_dataset, BATCH_SIZE)
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=dataset.num_classes,
        output_attentions=False,
        output_hidden_states=False
    )
    trainer = Trainer(model, train_loader, val_loader, device, BATCH_SIZE, LR, EPOCHS)
    trainer.load_model(f"../checkpoints/model_7labels_noMLP_0.0002_LR")
    train_accuracy = trainer.calculate_accuracy(train_loader, early_stop=100)
    print(train_accuracy)

