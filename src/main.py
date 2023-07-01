import torch
import pickle
import random
import pandas as pd
import os
import optuna
from dataset import DataSet
from bert_ect import EmotionClassifier
from Trainer import Trainer
from transformers import XLNetForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification
from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer, GPT2Tokenizer
from Model import TransformerECT
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 64
EPOCHS = 10
LR = 2e-5
DROPOUT = 0.1

OPTUNA = True

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_tokenizer.save_pretrained("./robert_tokenizer")

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_tokenizer.save_pretrained("./bert_tokenizer")

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

def check_gpu():
    # Check if GPU is available and being used
    print(torch.cuda.is_available())

    # Check the number of available GPUs
    print(torch.cuda.device_count())

    # Get the name of the current GPU
    print(torch.cuda.get_device_name(torch.cuda.current_device()))


def create_test_loader(path_to_data="../data/full_dataset/origin_data_proccessed.csv", test_p=0.15):
    if not os.path.exists("../data/unique_data/test_benchmark_data.csv"):
        raw_dataset = pd.read_csv(path_to_data)
        test_num = int(raw_dataset.shape[0] * test_p)
        random_indices = random.sample(range(len(raw_dataset)), test_num)
        benchmark_test_dataset = raw_dataset.loc[random_indices]
        reduced_dataset = raw_dataset.drop(random_indices)
        reduced_dataset.to_csv('../data/unique_data/raw_emotions_data.csv', index=False)
        benchmark_test_dataset.to_csv('../data/unique_data/test_benchmark_data.csv', index=False)
    else:
        benchmark_test_dataset = pd.read_csv("../data/unique_data/test_benchmark_data.csv")
    test_labels = benchmark_test_dataset['Emotion'].values.tolist()
    test_texts = benchmark_test_dataset['text'].values.tolist()

    roberta_tokenized_inputs = roberta_tokenizer.batch_encode_plus(test_texts, add_special_tokens=True,
                                                             return_attention_mask=True, pad_to_max_length=True,
                                                             truncation=True,
                                                             max_length=512,
                                                             return_tensors='pt')
    roberta_benchmark_test_dataset = TensorDataset(torch.tensor(roberta_tokenized_inputs['input_ids']),
                            torch.tensor(roberta_tokenized_inputs['attention_mask']),
                            torch.tensor(test_labels))
    
    bert_tokenized_inputs = bert_tokenizer.batch_encode_plus(test_texts, add_special_tokens=True,
                                                             return_attention_mask=True, pad_to_max_length=True,
                                                             truncation=True,
                                                             max_length=512,
                                                             return_tensors='pt')
    bert_benchmark_test_dataset = TensorDataset(torch.tensor(bert_tokenized_inputs['input_ids']),
                            torch.tensor(bert_tokenized_inputs['attention_mask']),
                            torch.tensor(test_labels))
    
    roberta_test_loader  = DataLoader(roberta_benchmark_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    bert_test_loader      = DataLoader(bert_benchmark_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return roberta_test_loader, bert_test_loader


def get_train_val_loaders_from_dataset(data_augmentation=False, is_robert=False):
    tokenizer = roberta_tokenizer if is_robert else bert_tokenizer
    dataset = DataSet(tokenizer, path_to_data='../data/unique_data/raw_emotions_data.csv')
    dataset.preprocessing_data(data_augmentation=data_augmentation)
    dataset.count_labels()
    # dataset.debug_rows()
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

    roberta_test_loader, bert_test_loader = create_test_loader()

    roberta_no_aug_train_loader, roberta_no_aug_val_loader, num_classes = get_train_val_loaders_from_dataset(is_robert=True)
    roberta_aug_train_loader, roberta_aug_val_loader, _ = get_train_val_loaders_from_dataset(data_augmentation=True, is_robert=True)

    bert_no_aug_train_loader, bert_no_aug_val_loader, num_classes = get_train_val_loaders_from_dataset()
    bert_aug_train_loader, bert_aug_val_loader, _ = get_train_val_loaders_from_dataset(data_augmentation=True)

    # optuna - find hyperparameters
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name="RoBerta", direction="maximize", sampler=sampler)
    study.optimize(lambda trial:
                   objective(trial,
                             device,
                             roberta_aug_train_loader,
                             roberta_aug_val_loader),
                   n_trials=100)
    
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print(" {}: {}".format(key, value))

    optuna.visualization.plot_param_importances(study)
    optuna.visualization.plot_contour(study, params=["n_units", "dropout"])
    optuna.visualization.plot_contour(study, params=["n_layers", "lr"])


    # model_names = [
    #     "NoAug_BERT_Inference", "NoAug_BERT_FT", "NoAug_BERT_Freeze_MLP", "NoAug_BERT_FT_MLP",
    #     "NoAug_RoBerta_Inference", "NoAug_RoBerta_FT", "NoAug_RoBerta_Freeze_MLP", "NoAug_RoBerta_FT_MLP",

    #     "Aug_BERT_Inference", "Aug_BERT_FT", "Aug_BERT_Freeze_MLP", "Aug_BERT_FT_MLP",
    #     "Aug_RoBerta_Inference", "Aug_RoBerta_FT", "Aug_RoBerta_Freeze_MLP", "Aug_RoBerta_FT_MLP"]



    # statistics_dict = {}

    # for mod_name in model_names:
    #     print(f"Starting Training of model {mod_name}: \n")
    #     statistics_dict[mod_name] = {}
    #     # Run Configurations
    #     feature_extracting = False if "FT" in mod_name else True
    #     mlp_enable = True if "MLP" in mod_name else False

    #     # Choose Model and data loaders
    #     # Choose between raw/augmented datasets
    #     if "RoBerta" in mod_name:
    #         is_roberta = True
    #         test_loader = roberta_test_loader
    #         if "NoAug" in mod_name:
    #             train_loader = roberta_no_aug_train_loader
    #             val_loader = roberta_no_aug_val_loader
    #         else:
    #             train_loader = roberta_aug_train_loader
    #             val_loader = roberta_aug_val_loader
    #     else:
    #         is_roberta = False
    #         test_loader = bert_test_loader
    #         if "NoAug" in mod_name:
    #             train_loader = bert_no_aug_train_loader
    #             val_loader = bert_no_aug_val_loader
    #         else:
    #             train_loader = bert_aug_train_loader
    #             val_loader = bert_aug_val_loader

    #     # Initiate model
    #     model = EmotionClassifier(num_classes, DROPOUT, is_roberta, feature_extracting=feature_extracting, mlp_enable=mlp_enable)
    #     trainer = Trainer(model, train_loader, val_loader, device, BATCH_SIZE, LR, EPOCHS)

    #     # If Inference - training is not needed
    #     if "Inference" not in mod_name:
    #         train_losses_list, val_acc_list = trainer.train()
    #         statistics_dict[mod_name]['Train Loss List'] = train_losses_list
    #         statistics_dict[mod_name]['Validation Accuracy List'] = val_acc_list

    #     # Calculate 
    #     accuracy = trainer.calculate_accuracy(test_loader)
    #     print(f"Model {mod_name} Has finised. Test accuracy is {accuracy}")
    #     statistics_dict[mod_name]['Test Accuracy'] = accuracy
    #     trainer.save_model(f"../checkpoints/{mod_name}.pth")
    
    # stat_path = "../History/models_stats.pickle"
    # with open(stat_path, 'wb') as file:
    #     pickle.dump(statistics_dict, file)