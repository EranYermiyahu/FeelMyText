import torch
import pickle
import random
import pandas as pd
from dataset import DataSet
from bert_ect import EmotionClassifier
from Trainer import Trainer
from transformers import XLNetForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification
from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer
from Model import TransformerECT
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 128
EPOCHS = 5
LR = 2e-5
DROPOUT = 0.1


def check_gpu():
    # Check if GPU is available and being used
    print(torch.cuda.is_available())

    # Check the number of available GPUs
    print(torch.cuda.device_count())

    # Get the name of the current GPU
    print(torch.cuda.get_device_name(torch.cuda.current_device()))


def create_test_loader(tokenizer, path_to_data="../data/full_dataset/origin_data_proccessed.csv", test_p=0.15):
    raw_dataset = pd.read_csv(path_to_data)
    test_num = int(raw_dataset.shape[0] * test_p)
    random_indices = random.sample(range(len(raw_dataset)), test_num)
    benchmark_test_dataset = raw_dataset.loc[random_indices]
    reduced_dataset = raw_dataset.drop(random_indices)
    reduced_dataset.to_csv('../data/full_dataset/raw_emotions_data.csv', index=False)

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


if __name__ == '__main__':
    # check_gpu()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.save_pretrained("./tokenizer")
    test_loader = create_test_loader(tokenizer)
    dataset = DataSet(tokenizer, path_to_data='../data/full_dataset/raw_emotions_data.csv')
    dataset.preprocessing_data(data_augmentation=False)
    dataset.count_labels()
    dataset.tokenized_inputs = tokenizer.batch_encode_plus(dataset.texts, add_special_tokens=True,
                                                             return_attention_mask=True, pad_to_max_length=True,
                                                             truncation=True, max_length=512, return_tensors='pt')
    train_dataset, val_dataset = dataset.split_train_val_data()
    train_loader, val_loader = dataset.create_data_loaders(train_dataset, val_dataset, BATCH_SIZE)

    accuracy_list = []
    model_names = ["NoAug_RoBerta_FT", "NoAug_RoBerta_FT_MLP"]
    model_name_inferance = "RoBerta_inferance"
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=dataset.num_classes,
        output_attentions=False,
        output_hidden_states=False,
        dropout=dp
    )
    # model = EmotionClassifier(dataset.num_classes, dropout, feature_extracting=True)
    trainer = Trainer(model, train_loader, val_loader, device, BATCH_SIZE, LR, EPOCHS)
    accuracy = trainer.calculate_accuracy(test_loader)
    print(accuracy)
    # trainer.train()
    # accuracy = trainer.calculate_accuracy(test_loader)
    # accuracy_list.append(accuracy)
    # train_accuracy = trainer.calculate_accuracy(train_loader, early_stop=10)
    # print(f"For dp {dp} the model accuracy on test is {accuracy} and on train is {train_accuracy}")
    # trainer.save_model(f"../checkpoints/model_dp_{dp}")




    history_name = f'../History/model_{model.model_name}_{dna_seq.viruses_num}viruses' + today.strftime('%d_%m_%Y-%H_%M') + '.pickle'
    with open(history_name, 'wb') as file:
        pickle.dump(fit.history, file)


    # dataset.data.to_csv('filtered_csv.csv', index=False)





# input_dim = dataset.vocab_size
# n_labels = dataset.num_classes
# hidden_dim = 256
# num_layers = 6
# num_heads = 8
# dropout = 0.3
# model = TransformerECT(input_dim, n_labels, hidden_dim, num_layers, num_heads, dropout)













