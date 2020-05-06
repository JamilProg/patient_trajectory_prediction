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
        self.hidden = nn.Linear(ARGS.inputdim, 512)
        # Hidden to hidden layer
        self.hidden2 = nn.Linear(512, 256)
        # Hidden to output layer
        self.output = nn.Linear(256, ARGS.numberOfOutputCodes)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.hidden2(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
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
    model.load_state_dict(torch.load(ARGS.inputModel))
    model.eval()
    predictions = np.zeros((len(X_test), ARGS.numberOfOutputCodes))
    i = 0
    for test_batch in tqdm(test_loader,disable = True):
        # batch_prediction = model(test_batch).detach().cpu().numpy()
        batch_prediction = model(test_batch)
        predictions[i * BATCH_SIZE:(i+1) * BATCH_SIZE, :] = batch_prediction
        i+=1
    

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
