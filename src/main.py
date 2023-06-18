import torch
import pandas as pd
from dataset import DataSet

def check_gpu():
    # Check if GPU is available and being used
    print(torch.cuda.is_available())

    # Check the number of available GPUs
    print(torch.cuda.device_count())

    # Get the name of the current GPU
    print(torch.cuda.get_device_name(torch.cuda.current_device()))


if __name__ == '__main__':
    # check_gpu()
    dataset = DataSet()
    dataset.remove_unclear_samples()
    dataset.add_emotion_label()
    dataset.tokenizer()
    dataset.split_train_test_val_data()
    # dataset.data.to_csv('filtered_csv.csv', index=False)