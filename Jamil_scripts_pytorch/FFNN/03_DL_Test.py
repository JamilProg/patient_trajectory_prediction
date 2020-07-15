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
from sklearn.metrics import roc_auc_score as roc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(ARGS.inputdim, ARGS.hiddenDimSize)
        self.hidden2 = nn.Linear(ARGS.hiddenDimSize, ARGS.numberOfOutputCodes)
        # Define sigmoid activation and softmax output
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # DropOut
        self.dropout = nn.Dropout(p=ARGS.dropOut)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.hidden2(x)
        return x


def test():
    # Load the test data
    X_test = pickle.load(open(ARGS.Xinputdata, 'rb'))
    Y_test = pickle.load(open(ARGS.Yinputdata, 'rb'))
    criterion = nn.BCEWithLogitsLoss()
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
        shuffle=False)

    # Load the model
    model = Network()
    model.load_state_dict(torch.load(ARGS.inputModel, map_location=torch.device('cpu')))
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    total = 0
    correct = 0
    model.eval()

    # validation loss
    loss_values = []
    itr_ctr = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            itr_ctr += 1
            data, target = Variable(data), Variable(target)
            net_out = model(data)
            loss = criterion(net_out, target)
            loss_values.append(loss)

    print("Validation loss :", np.mean(loss_values))

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

    # Recalls
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

    # AUC score
    AUC_list = list()
    YTRUE = None
    YPROBA = None
    for data in test_loader:
        x, labels = data
        x, labels = Variable(x), Variable(labels)
        outputs = model(x).detach().numpy()
        labels = labels.detach().numpy()
        # roc_score=roc(labels, outputs, average='micro', multi_class='ovr')
        # AUC_list.append(roc_score)
        for batch_true, batch_prob in zip(labels, outputs):
            YTRUE = np.concatenate((YTRUE, [batch_true]), axis=0) if YTRUE is not None else [batch_true]
            YPROBA = np.concatenate((YPROBA, [batch_prob]), axis=0) if YPROBA is not None else [batch_prob]
    # ROC_avg_score=sum(AUC_list)/len(AUC_list)
    ROC_avg_score=roc(YTRUE, YPROBA, average='micro', multi_class='ovr')
    print("ROC Average Score:", ROC_avg_score)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Xinputdata', type=str, default='X-test.data', metavar='<visit_file>')
    parser.add_argument('--Yinputdata', type=str, default='Y-test.data', metavar='<visit_file>')
    parser.add_argument('--inputModel', type=str, default='model_output.pt', metavar='<visit_file>')
    parser.add_argument('--batchSize', type=int, default=100, help='Batch size.')
    parser.add_argument('--hiddenDimSize', type=int, default=10000, help='Number of neurons in hidden layer.')
    parser.add_argument('--dropOut', type=float, default=0.5, help='Dropout rate.')

    ARGStemp = parser.parse_args()
    return ARGStemp


if __name__ == '__main__':
    global ARGS
    ARGS = parse_arguments()
    test()
