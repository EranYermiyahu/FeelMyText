
import torch
from dataset import DataSet

def check_gpu():
    # Check if GPU is available and being used
    print(torch.cuda.is_available())

    # Check the number of available GPUs
    print(torch.cuda.device_count())

    # Get the name of the current GPU
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

if __name__ == '__main__':
    check_gpu()

    dataset = DataSet()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
