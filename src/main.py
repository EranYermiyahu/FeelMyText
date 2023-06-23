import torch
import pandas as pd
from dataset import DataSet
from bert_ect import EmotionClassifier
from Trainer import Trainer
from Model import TransformerECT
from torch.utils.data import TensorDataset, DataLoader

def check_gpu():
    # Check if GPU is available and being used
    print(torch.cuda.is_available())

    # Check the number of available GPUs
    print(torch.cuda.device_count())

    # Get the name of the current GPU
    print(torch.cuda.get_device_name(torch.cuda.current_device()))


BATCH_SIZE = 128
EPOCHS = 50
LR_LIST = [2e-3, 2e-4, 2e-5]
DROPOUT = 0.3


if __name__ == '__main__':
    check_gpu()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset = DataSet()
    dataset.remove_unclear_samples()
    dataset.add_emotion_label()
    dataset.print_labels_count()
    dataset.tokenizer()
    train_dataset, val_dataset, test_dataset = dataset.split_train_test_val_data()
    # Create data loaders for train, test, and validation sets
    train_loader, test_loader, val_loader = dataset.create_data_loaders(train_dataset, val_dataset, test_dataset, BATCH_SIZE)

    input_dim = dataset.vocab_size
    n_labels = dataset.num_classes
    hidden_dim = 256
    num_layers = 6
    num_heads = 8
    dropout = 0.3

    # model = TransformerECT(input_dim, n_labels, hidden_dim, num_layers, num_heads, dropout)
    accuracy_list = []
    for LR in LR_LIST:
        model = EmotionClassifier(dataset.num_classes, dropout=DROPOUT)
        trainer = Trainer(model, train_loader, val_loader, device, BATCH_SIZE, LR)
        trainer.train(EPOCHS)
        accuracy = trainer.calculate_accuracy(test_loader)
        accuracy_list.append(accuracy)
        print(f"For LR {LR} the model accuracy on test is {accuracy}")




    # dataset.data.to_csv('filtered_csv.csv', index=False)