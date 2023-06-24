import torch
import pickle
import random
import pandas as pd
import os
from dataset import DataSet
from bert_ect import EmotionClassifier
from Trainer import Trainer
from transformers import XLNetForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification
from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer
from Model import TransformerECT
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 128
EPOCHS = 10
LR = 2e-5
DROPOUT = 0.1
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer.save_pretrained("./tokenizer")

def check_gpu():
    # Check if GPU is available and being used
    print(torch.cuda.is_available())

    # Check the number of available GPUs
    print(torch.cuda.device_count())

    # Get the name of the current GPU
    print(torch.cuda.get_device_name(torch.cuda.current_device()))


def create_test_loader(path_to_data="../data/full_dataset/origin_data_proccessed.csv", test_p=0.15):
    if not os.path.exists("../data/full_dataset/test_benchmark_data.csv"):
        raw_dataset = pd.read_csv(path_to_data)
        test_num = int(raw_dataset.shape[0] * test_p)
        random_indices = random.sample(range(len(raw_dataset)), test_num)
        benchmark_test_dataset = raw_dataset.loc[random_indices]
        reduced_dataset = raw_dataset.drop(random_indices)
        reduced_dataset.to_csv('../data/full_dataset/raw_emotions_data.csv', index=False)
        benchmark_test_dataset.to_csv('../data/full_dataset/test_benchmark_data.csv', index=False)
    else:
        benchmark_test_dataset = pd.read_csv("../data/full_dataset/test_benchmark_data.csv")
    test_labels = benchmark_test_dataset['Emotion'].values.tolist()
    test_texts = benchmark_test_dataset['text'].values.tolist()
    tokenized_inputs = tokenizer.batch_encode_plus(test_texts, add_special_tokens=True,
                                                             return_attention_mask=True, pad_to_max_length=True,
                                                             truncation=True,
                                                             max_length=512,
                                                             return_tensors='pt')
    benchmark_test_dataset = TensorDataset(torch.tensor(tokenized_inputs['input_ids']),
                            torch.tensor(tokenized_inputs['attention_mask']),
                            torch.tensor(test_labels))
    return DataLoader(benchmark_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def get_train_val_loaders_from_dataset(data_augmentation=False):
    dataset = DataSet(tokenizer, path_to_data='../data/full_dataset/raw_emotions_data.csv')
    dataset.preprocessing_data(data_augmentation=data_augmentation)
    dataset.count_labels()
    dataset.tokenized_inputs = tokenizer.batch_encode_plus(dataset.texts, add_special_tokens=True,
                                                           return_attention_mask=True, pad_to_max_length=True,
                                                           truncation=True, max_length=512, return_tensors='pt')
    train_dataset, val_dataset = dataset.split_train_val_data()
    train_loader, val_loader = dataset.create_data_loaders(train_dataset, val_dataset, BATCH_SIZE)
    return train_loader, val_loader, dataset.num_classes

if __name__ == '__main__':
    # check_gpu()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    test_loader = create_test_loader()
    no_aug_train_loader, no_aug_val_loader, num_classes = get_train_val_loaders_from_dataset()
    aug_train_loader, aug_val_loader, _ = get_train_val_loaders_from_dataset(data_augmentation=True)

    model_names = ["NoAug_RoBerta_Inference", "NoAug_RoBerta_FT", "NoAug_RoBerta_Freeze_MLP", "NoAug_RoBerta_FT_MLP",
                   "Aug_RoBerta_Inference", "Aug_RoBerta_FT", "Aug_RoBerta_Freeze_MLP", "Aug_RoBerta_FT_MLP"]
    statistics_dict = {}

    for mod_name in model_names:
        feature_extracting = False if "FT" in mod_name else True
        mlp_enable = True if "MLP" in mod_name else False
        model = EmotionClassifier(num_classes, DROPOUT, feature_extracting=feature_extracting, mlp_enable=mlp_enable)

        if "NoAug" in mod_name:
            trainer = Trainer(model, no_aug_train_loader, no_aug_val_loader, device, BATCH_SIZE, LR, EPOCHS)
        else:
            trainer = Trainer(model, aug_train_loader, aug_val_loader, device, BATCH_SIZE, LR, EPOCHS)

        if "Inference" not in mod_name:
            train_losses_list, val_acc_list = trainer.train()
            statistics_dict[mod_name]['Train Loss List'] = train_losses_list
            statistics_dict[mod_name]['Validation Accuracy List'] = val_acc_list

        accuracy = trainer.calculate_accuracy(test_loader)
        statistics_dict[mod_name]['Test Accuracy'] = accuracy
        trainer.save_model(f"../checkpoints/{mod_name}.pth")
        stat_path = f"../History/{mod_name}_stats.pickle"
        with open(stat_path, 'wb') as file:
            pickle.dump(statistics_dict, file)



    # dataset.data.to_csv('filtered_csv.csv', index=False)





# input_dim = dataset.vocab_size
# n_labels = dataset.num_classes
# hidden_dim = 256
# num_layers = 6
# num_heads = 8
# dropout = 0.3
# model = TransformerECT(input_dim, n_labels, hidden_dim, num_layers, num_heads, dropout)













