#!/usr/bin/python
import pickle
import math
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dt
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm_notebook as tqdm


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        # Hidden layer
        # self.hidden = nn.Linear(ARGS.inputdim, 8192)
        # self.hidden2 = nn.Linear(8192, ARGS.numberOfOutputCodes)
        self.hidden = nn.Linear(ARGS.inputdim, 10000)
        self.hidden2 = nn.Linear(10000, ARGS.numberOfOutputCodes)
        # Define sigmoid activation and softmax output
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        return x


def test():
    # Load the test data
    X_test = pickle.load(open(ARGS.Xinputdata, 'rb'))
    Y_test = pickle.load(open(ARGS.Yinputdata, 'rb'))
    ARGS.inputdim = len(X_test[0])
    ARGS.numberOfOutputCodes = len(Y_test[0])
    print("X_test of len:", len(X_test), "and Y_test of len:", len(Y_test))
    print("Samples of X's len:", len(X_test[0]), "and samples of Y's len:", len(Y_test[0]))
    tensor_x = torch.Tensor(np.array(X_test)) # transform to torch tensor
    print("X_dataset_shape=",tensor_x.shape)
    tensor_y = torch.Tensor(np.array(Y_test))
    print("Y_dataset_shape=",tensor_y.shape)
    dataset = dt.TensorDataset(tensor_x, tensor_y) # create your dataset
    test_size = int(len(dataset))
    print("test_size =", test_size)
    batchsize = ARGS.batchSize
    test_loader = dt.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True)

    # Load the model
    model = Network()
    model.load_state_dict(torch.load(ARGS.inputModel, map_location=torch.device('cpu')))
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    total = 0
    correct = 0
    model.eval()

    P = list()
    R = list()
    # Precisions
    for i in range(1,4):
        for data in test_loader:
            x, labels = data
            outputs = model(x) # output is a tensor of size [BATCHSIZE][ARGS.numberOfOutputCodes]
            # _, predicted = torch.max(outputs.data, 1)
            _, predicted = torch.topk(outputs.data, i)
            for y_predlist, y in zip(predicted, labels):
                for y_pred in y_predlist:
                    total += 1
                    if y[y_pred] == 1:
                        correct += 1

        precision = correct / total
        print("P@", i, "=", precision)
        P.append(precision)
        correct = 0
        total = 0

    # Number of diagnostic for each sample (mean of 12 codes, max of 30 codes, R@10 - R@20 - R@30 seems appropriate)
    total_true_list = list()
    for data in test_loader:
        x, labels = data
        outputs = model(x)
        for y in labels :
            total_true = 0
            for val in y :
                if val == 1:
                    total_true += 1
            total_true_list.append(total_true)

    # recall
    for i in range(10,40,10):
        total_true_list_cpy = list(total_true_list)
        for data in test_loader:
            x, labels = data
            outputs = model(x)
            _, predicted = torch.topk(outputs.data, i)
            for y_predlist, y in zip(predicted, labels):
                total += total_true_list_cpy.pop(0)
                for y_pred in y_predlist:
                    if y[y_pred] == 1:
                        correct += 1

        recall = correct / total
        print("R@", i, "=", recall)
        R.append(recall)
        correct = 0
        total = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Xinputdata', type=str, default='X-test.data', metavar='<visit_file>')
    parser.add_argument('--Yinputdata', type=str, default='Y-test.data', metavar='<visit_file>')
    parser.add_argument('--inputModel', type=str, default='model_output.pt', metavar='<visit_file>')
    parser.add_argument('--batchSize', type=int, default=100, help='Batch size')
    ARGStemp = parser.parse_args()
    return ARGStemp


if __name__ == '__main__':
    global ARGS
    ARGS = parse_arguments()
    test()
