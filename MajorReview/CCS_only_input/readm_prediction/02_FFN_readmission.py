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
from sklearn.metrics import roc_auc_score as roc
import gc

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # ARGS.inputdim = ARGS.numberOfInputCUIInts + ARGS.numberOfInputCCSInts if ARGS.withCCS else ARGS.numberOfInputCUIInts
        ARGS.inputdim = ARGS.numberOfInputCCSInts
        self.fc1 = nn.Linear(ARGS.inputdim, ARGS.hiddenDimSize)
        self.fc2 = nn.Linear(ARGS.hiddenDimSize, 1)
        self.relu = nn.ReLU()
        self.sigmo = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Prop input through FFN
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.sigmo(x)
        return x

class my_dataset(dt.Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label
    def __getitem__(self, index):
        return self.data[index],self.label[index]
    def __len__(self):
        return len(self.data)


def load_tensors():
    #-----------------------------------------
    # pickle input map - each entry is a pair (subject_id, [(hadm_id,admittime, [CUIsvector], [CCSsvector])]
    # notesVectors_trainMapX = pickle.load(open(ARGS.inputFileNotes, 'rb'))
    subjecttoadm_map = pickle.load(open(ARGS.inputdata, 'rb'))
    setOfDistinctCUIs = set()
    setOfDistinctCCSs = set()
    cuitoint = dict()
    ccstoint = dict()
    for subject in subjecttoadm_map.keys():
        patientData = subjecttoadm_map[subject]
        for ithAdmis in patientData:
            for CUIcode in ithAdmis[2]:
                setOfDistinctCUIs.add(CUIcode)
            for CCScode in ithAdmis[3]:
                setOfDistinctCCSs.add(CCScode)
    for i, cui in enumerate(setOfDistinctCUIs):
        cuitoint[cui] = i
    for i, ccs in enumerate(setOfDistinctCCSs):
        ccstoint[ccs] = i
    print("-> " + str(len(subjecttoadm_map)) + " patients' CUI notes and CCS codes at dimension 0 for file: "+ ARGS.inputdata)
    ARGS.numberOfInputCUIInts = len(setOfDistinctCUIs)
    ARGS.numberOfInputCCSInts = len(setOfDistinctCCSs)
    # -------------------------------------------
    ARGS.numberOfOutputCodes = 1
    print('Remaining patients:', len(subjecttoadm_map))
    # Convert everything to list of list of list (patient x admission x CUInote_vector/mortality
    # to ease the manipulation in batches
    vectors_trainListX = []
    readmission_trainListY = []
    maxSeqLength = 0
    for pID, adList in subjecttoadm_map.items():
        for i, adm in enumerate(adList):
            if i+1 == len(adList):
                # Avoid the last admission
                continue
            one_hot_CUI = [0] * ARGS.numberOfInputCUIInts
            one_hot_CCS = [0] * ARGS.numberOfInputCCSInts
            for cui_int in adm[2]:
                one_hot_CUI[cuitoint[cui_int]] = 1
            for ccs_int in adm[3]:
                one_hot_CCS[ccstoint[ccs_int]] = 1
            # one_hot_X = one_hot_CUI + one_hot_CCS if ARGS.withCCS else one_hot_CUI
            one_hot_X = one_hot_CCS
            vectors_trainListX.append(one_hot_X)
            # compute patient readmission
            within30days = [1]
            if (adList[i+1][1] - adm[4]).days > 30:
                within30days[0] = 0
            readmission_trainListY.append(within30days)

    # Randomize in dimension 0 (patients order) keeping the notes and mortality in sync
    mapIndexPosition = list(zip(vectors_trainListX, readmission_trainListY))
    random.shuffle(mapIndexPosition)
    vectors_trainListX, readmission_trainListY = zip(*mapIndexPosition)

    return vectors_trainListX, readmission_trainListY


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def train():
    data_x, data_y = load_tensors()

    print("Available GPU :", torch.cuda.is_available())
    torch.cuda.set_device(1)

    sizedata = len(data_x)
    print("Data of size:", sizedata)
    # Split dataset into 5 sub-datasets
    splitted_x = list(split(data_x, 5))
    splitted_y = list(split(data_y, 5))
    k = ARGS.kFold

    del data_x
    del data_y
    gc.collect()
    print("Memory freed")

    AUC_folds = []
    for ind_i in range(0, k):
        # Prepare X_train Y_train X_test Y_test
        X_test = splitted_x[ind_i]
        Y_test = splitted_y[ind_i]
        # Deep copy, otherwise iteration problem
        copysplitX = list(splitted_x)
        copysplitY = list(splitted_y)
        del copysplitX[ind_i]
        del copysplitY[ind_i]
        X_train = copysplitX
        Y_train = copysplitY
        model = Network().cuda()

        with torch.cuda.device(1):
            # Hyperparameters :
            epochs = ARGS.nEpochs
            batchsize = ARGS.batchSize
            learning_rate = ARGS.lr
            log_interval = 2
            criterion = nn.BCEWithLogitsLoss()
            # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            # Data loader0
            tensor_x = torch.ByteTensor(np.concatenate([np.array(plist, dtype=np.uint8) for plist in X_train]).tolist()).cuda()
            tensor_y = torch.ByteTensor(np.concatenate([np.array(plist, dtype=np.uint8) for plist in Y_train]).tolist()).cuda()
            print("Shape X:", tensor_x.size(), "Shape Y:", tensor_y.size())
            dataset = dt.TensorDataset(tensor_x, tensor_y)  # create your dataset
            print("TRAINLOADER")
            train_loader = dt.DataLoader(
                dataset,
                batch_size=batchsize,
                shuffle=True)
            gc.collect()
            
            # Training
            for epoch in range(epochs):
                for batch_idx, (data, target) in enumerate(train_loader):
                    # data, target = data.cuda(), target.cuda()
                    data, target = Variable(data.float()), Variable(target.float())
                    optimizer.zero_grad()
                    net_out = model(data)
                    loss = criterion(net_out, target)
                    loss.backward()
                    optimizer.step()
                    if batch_idx % log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: '.format(
                                epoch, batch_idx * len(data), len(train_loader.dataset),
                                       100. * batch_idx / len(train_loader)))
                        print(loss.data)

            print("Training done - deleting variables and starting test")
            dataset = None
            train_loader = None
            tensor_x = None
            tensor_y = None
            arrX = None
            arrY = None
            numplist = None
            copysplitX = None
            copysplitY = None
            del dataset
            del train_loader
            del tensor_x
            del tensor_y
            del arrX
            del arrY
            del numplist
            del copysplitX
            del copysplitY
            gc.collect()

            # Test loader
            tensor_x = torch.Tensor(np.array(X_test).tolist()).cuda()  # transform to torch tensor
            tensor_y = torch.Tensor(np.array(Y_test).tolist()).cuda()
            
            dataset = dt.TensorDataset(tensor_x, tensor_y)  # create your dataset
            test_loader = dt.DataLoader(
                dataset,
                batch_size=batchsize,
                shuffle=False)

            # Testing
            model.eval()
            
            #AUROC 
            AUC_list = list()
            YTRUE = None
            YPROBA = None
            for (data, targets) in test_loader:
                x, labels = Variable(data.float()), Variable(targets.float())
                outputs = model(x)
                outputs = outputs.detach().cpu().numpy()
                x = x.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                for batch_x, batch_true, batch_prob in zip(x, labels, outputs):
                    YTRUE = np.concatenate((YTRUE, [batch_true]), axis=0) if YTRUE is not None else [batch_true]
                    YPROBA = np.concatenate((YPROBA, [batch_prob]), axis=0) if YPROBA is not None else [batch_prob]
            ROC_avg_score=roc(YTRUE, YPROBA, average='macro', multi_class='ovo')
            print("ROC Average Score:", ROC_avg_score)
            AUC_folds.append(ROC_avg_score)

    # Output score of each fold + average
    print("AUC_scores: ", AUC_folds)
    print("Mean: ", sum(AUC_folds)/len(AUC_folds))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdata', type=str, default='prepared_data.npz', metavar='<visit_file>')
    parser.add_argument('--hiddenDimSize', type=int, default=50, help='Size of FFN hidden layer')
    parser.add_argument('--batchSize', type=int, default=100, help='Batch size.')
    parser.add_argument('--nEpochs', type=int, default=100, help='Number of training iterations.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--dropOut', type=float, default=0.5, help='GRU Dropout.')
    parser.add_argument('--kFold', type=int, default=5, help='K value (int) of K-fold cross-validation.')
    parser.add_argument('--withCCS', help='add CCS feature in input.')

    ARGStemp = parser.parse_args()
    return ARGStemp


if __name__ == '__main__':
    global ARGS
    ARGS = parse_arguments()
    train()
