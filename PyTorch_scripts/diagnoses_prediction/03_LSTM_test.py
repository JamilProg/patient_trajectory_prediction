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


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_classes = ARGS.numberOfOutputCodes
        self.num_layers = ARGS.numLayers
        self.hidden_size = ARGS.hiddenDimSize
        self.lstm = nn.LSTM(input_size=ARGS.inputdim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=ARGS.dropOut)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, hidden):
        # Prop input through LSTM
        bs = x.size(0)
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(bs,-1, self.hidden_size)
        out = self.fc(lstm_out)
        # out = out.view(ARGS.batchSize, -1)
        # out = out[:,-1]
        return out, hidden

    def init_hidden(self):
        # weight = next(self.parameters()).data
        # hidden = (weight.new(self.num_layers, ARGS.batchSize, self.hidden_size).zero_(),weight.new(self.num_layers, ARGS.batchSize, self.hidden_size).zero_())
        h_0 = torch.randn(self.num_layers, ARGS.batchSize, self.hidden_size)
        c_0 = torch.randn(self.num_layers, ARGS.batchSize, self.hidden_size)
        hidden = (Variable(h_0), Variable(c_0))
        # hidden = (torch.randn(self.num_layers, ARGS.batchSize, self.hidden_size),torch.randn(self.num_layers, ARGS.batchSize, self.hidden_size))
        return hidden


def test():
    # Load the test data
    X_test = pickle.load(open(ARGS.Xinputdata, 'rb'))
    Y_test = pickle.load(open(ARGS.Yinputdata, 'rb'))
    ARGS.inputdim = len(X_test[0][0])
    ARGS.numberOfOutputCodes = len(Y_test[0][0])
    print("Dataset with :", len(X_test), "patients. Y:", len(Y_test))
    print("Each patient has :", len(X_test[0]), "admissions. Y:", len(Y_test[0]))
    X_test=np.array([np.array([np.array(unelist, dtype=np.uint8) for unelist in xi]) for xi in X_test])
    Y_test=np.array([np.array([np.array(unelist, dtype=np.uint8) for unelist in xi]) for xi in Y_test])
    tensor_x = torch.from_numpy(X_test)
    tensor_y = torch.from_numpy(Y_test)
    print("X_dataset_shape=",tensor_x.shape)
    print("Y_dataset_shape=",tensor_y.shape)
    dataset = dt.TensorDataset(tensor_x, tensor_y) # create your dataset
    batchsize = ARGS.batchSize
    test_loader = dt.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False)

    # Load the model
    model = Network()
    model.load_state_dict(torch.load(ARGS.inputModel))
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    total = 0
    correct = 0
    model.eval()
    h = model.init_hidden()
    P = list()
    R = list()
    # Precisions
    for i in range(1,4):
        for (data, targets) in test_loader:
            x, labels = Variable(data.float()), Variable(targets.float())
            # output is a tensor of size [BATCHSIZE][#ADMISSIONS][ARGS.numberOfOutputCodes]
            if (x.size(0) != ARGS.batchSize):
                continue
            outputs, h = model(x, h)
            _, predicted = torch.topk(outputs.data, i)
            for y_predlist_adm, y_adm in zip(predicted, targets):
                for y_predlist, y in zip(y_predlist_adm, y_adm):
                    # If y is a tensor with only zeros (padding), break this loop
                    if torch.max(y) != 1 :
                        break
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
    h = model.init_hidden()
    for (data, targets) in test_loader:
        x, labels = Variable(data.float()), Variable(targets.float())
        if (x.size(0) != ARGS.batchSize):
            continue
        outputs, h = model(x, h)
        for y_adm in targets :
            for y in y_adm :
                if torch.max(y) != 1 :
                    break
                total_true = 0
                for val in y :
                    if val == 1:
                        total_true += 1
                total_true_list.append(total_true)

    # recall
    h = model.init_hidden()
    for i in range(10,40,10):
        total_true_list_cpy = list(total_true_list)
        for (data, targets) in test_loader:
            x, labels = Variable(data.float()), Variable(targets.float())
            if (x.size(0) != ARGS.batchSize):
                continue
            outputs, h = model(x,h)
            _, predicted = torch.topk(outputs.data, i)
            for y_predlist_adm, y_adm in zip(predicted, targets):
                for y_predlist, y in zip(y_predlist_adm, y_adm):
                    if torch.max(y) != 1 :
                        break
                    total += total_true_list_cpy.pop(0)
                    for y_pred in y_predlist:
                        if y[y_pred] == 1:
                            correct += 1

        recall = correct / total
        print("R@", i, "=", recall)
        R.append(recall)
        correct = 0
        total = 0

    # AUROC
    YTRUE = None
    YPROBA = None
    h = model.init_hidden()
    for (data, targets) in test_loader:
        x, labels = Variable(data.float()), Variable(targets.float())
        if x.size(0) != ARGS.batchSize:
            continue
        outputs, h = model(x, h)
        outputs = outputs.detach().cpu().numpy()
        x = x.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        for batch_true, batch_prob in zip(labels, outputs):
            for adm_true, adm_prob in zip(batch_true, batch_prob):
                if torch.max(torch.from_numpy(adm_true)) != 1:
                    break
                YTRUE = np.concatenate((YTRUE, [adm_true]), axis=0) if YTRUE is not None else [adm_true]
                YPROBA = np.concatenate((YPROBA, [adm_prob]), axis=0) if YPROBA is not None else [adm_prob]
    ROC_avg_score = roc(YTRUE, YPROBA, average='micro', multi_class='ovr')
    print("ROC Average Score:", ROC_avg_score)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Xinputdata', type=str, default='X-test.data', metavar='<visit_file>')
    parser.add_argument('--Yinputdata', type=str, default='Y-test.data', metavar='<visit_file>')
    parser.add_argument('--inputModel', type=str, default='model_output.pt', metavar='<visit_file>')
    parser.add_argument('--batchSize', type=int, default=10, help='Batch size')
    parser.add_argument('--hiddenDimSize', type=int, default=200, help='Size of LSTM hidden layer')
    parser.add_argument('--numLayers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--dropOut', type=float, default=0, help='LSTM Dropout.')
    ARGStemp = parser.parse_args()
    return ARGStemp


if __name__ == '__main__':
    global ARGS
    ARGS = parse_arguments()
    test()
