import torch
import pandas as pd
from dataset import DataSet
from bert_ect import EmotionClassifier
from Trainer import Trainer
from torch.utils.data import TensorDataset, DataLoader

def check_gpu():
    # Check if GPU is available and being used
    print(torch.cuda.is_available())

    # Check the number of available GPUs
    print(torch.cuda.device_count())

    # Get the name of the current GPU
    print(torch.cuda.get_device_name(torch.cuda.current_device()))


BATCH_SIZE = 50
EPOCHS = 5


if __name__ == '__main__':
    # check_gpu()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = DataSet()
    dataset.remove_unclear_samples()
    dataset.add_emotion_label()
    dataset.tokenizer()
    train_dataset, val_dataset, test_dataset = dataset.split_train_test_val_data()
    # Create data loaders for train, test, and validation sets
    train_loader, test_loader, val_loader = dataset.create_data_loaders(train_dataset, val_dataset, test_dataset, BATCH_SIZE)
    model = EmotionClassifier(dataset.num_classes)
    trainer = Trainer(model, train_loader, val_loader, device, BATCH_SIZE)
    trainer.train(EPOCHS)




    # dataset.data.to_csv('filtered_csv.csv', index=False)