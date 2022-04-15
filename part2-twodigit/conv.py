import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

# check cuda is available
torch.cuda.is_available()

batch_size = 128
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
# function that move data/model to GPU
def to_device(data,device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    elif isinstance(data,dict):
        res = {}
        for k,v in data.items():
            res[k] = to_device(v,device=device)
    else:
        raise TypeError("Invalid Type of device transfer")


class DeviceDataLoader():
    #Wrap a dataloader to move data to a device
    def __init__(self,dl,device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        # Yield a batch of data after moving it to device
        for b in self.dl:
            yield to_device(b,self.device)
    def  __len__(self):
        #Number of batch
        return len(self.dl)


class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        # TODO initialize model layers here
        self.conv2d_1 = nn.Conv2d(1, 32, (3, 3))
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d((2, 2))
        self.conv2d_2 = nn.Conv2d(32, 64, (3, 3))
        self.flatten = Flatten()
        self.linear1 = nn.Linear(2880, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(64, 20)

    def forward(self, x):

        # TODO use model layers to predict the two digits
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)

        out_first_digit = x[:, :10]
        out_second_digit = x[:, 10:]
        return out_first_digit, out_second_digit
def batchify_data1(x_data, y_data, batch_size):
    """Takes a set of data points and labels and groups them into batches."""
    # Only take batch_size chunks (i.e. drop the remainder)
    device = get_default_device()
    N = int(len(x_data) / batch_size) * batch_size
    batches = []
    for i in range(0, N, batch_size):
        batches.append({
            'x': torch.tensor(x_data[i:i + batch_size],
                              dtype=torch.float32).to(device),
            'y': torch.tensor([y_data[0][i:i + batch_size],
                               y_data[1][i:i + batch_size]],
                               dtype=torch.int64).to(device)
        })
    return batches
def main():
    device = get_default_device()
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)
    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Move data to GPU
    # X_train = torch.tensor(X_train)
    # X_test = torch.tensor(X_test)
    # y_train = torch.tensor(y_train)
    # y_test = torch.tensor(y_test)
    #
    # X_train, X_test = X_train.to(device), X_test.to(device)
    # y_train, y_test = y_train.to(device), y_test.to(device)

    #X_train, X_test = to_device(X_train,device), to_device(X_test,device)
    #y_train, y_test = y_train.to(device), y_test.to(device)

    # Split dataset into batches
    train_batches = batchify_data1(X_train, y_train, batch_size)
    dev_batches = batchify_data1(X_dev, y_dev, batch_size)
    test_batches = batchify_data1(X_test, y_test, batch_size)


    #train_batches =DeviceDataLoader(train_batches,device)
    #dev_batches = DeviceDataLoader(dev_batches,device)
    #test_batches = DeviceDataLoader(test_batches,device)
    #


    # Load model
    input_dimension = img_rows * img_cols
    #model = CNN(input_dimension) # TODO add proper layers to CNN class above
    model = CNN(input_dimension).to(device)
    #model = to_device(model,device) # Move model to device
    # Train
    train_model(train_batches, dev_batches, model)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
