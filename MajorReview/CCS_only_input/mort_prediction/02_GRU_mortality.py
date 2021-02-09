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

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        #ARGS.inputdim = ARGS.numberOfInputCUIInts + ARGS.numberOfInputCCSInts if ARGS.withCCS else ARGS.numberOfInputCUIInts
        ARGS.inputdim = ARGS.numberOfInputCCSInts
        self.num_classes = 3
        self.num_layers = ARGS.numLayers
        self.hidden_size = ARGS.hiddenDimSize
        self.gru = nn.GRU(input_size=ARGS.inputdim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=ARGS.dropOut)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, hidden):
        # Prop input through GRU
        bs = x.size(0)
        gru_out, hidden = self.gru(x, hidden)
        gru_out = gru_out.contiguous().view(bs,-1, self.hidden_size)
        out = self.fc(gru_out)
        return out, hidden

    def init_hidden(self):
        h_0 = torch.randn(self.num_layers, ARGS.batchSize, self.hidden_size).cuda()
        hidden = Variable(h_0)
        return hidden


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
    subjecttodeath_map = pickle.load(open(ARGS.inputdata2, 'rb'))
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
    ARGS.numberOfOutputCodes = 3
    print('Remaining patients:', len(subjecttoadm_map))
    # Convert everything to list of list of list (patient x admission x CUInote_vector/mortality
    # to ease the manipulation in batches
    vectors_trainListX = []
    mortality_trainListY = []
    maxSeqLength = 0
    for pID, adList in subjecttoadm_map.items():
        sequence_X = []
        sequence_Y = []
        if len(adList) > maxSeqLength:
            maxSeqLength = len(adList)
        for i, adm in enumerate(adList):
            one_hot_CUI = [0] * ARGS.numberOfInputCUIInts
            one_hot_CCS = [0] * ARGS.numberOfInputCCSInts
            for cui_int in adm[2]:
                one_hot_CUI[cuitoint[cui_int]] = 1
            for ccs_int in adm[3]:
                one_hot_CCS[ccstoint[ccs_int]] = 1
            # one_hot_X = one_hot_CUI + one_hot_CCS if ARGS.withCCS else one_hot_CUI
            one_hot_X = one_hot_CCS
            sequence_X.append(one_hot_X)
            # compute mortality one-hot
            mortality_onehot = [0] * 3
            if (subjecttodeath_map[pID] - adm[4]).days <= 0:
                mortality_onehot[0] = 1
            elif (subjecttodeath_map[pID] - adm[4]).days <= 30:
                mortality_onehot[1] = 1
            else:
                mortality_onehot[2] = 1
            sequence_Y.append(mortality_onehot)
        vectors_trainListX.append(sequence_X)
        mortality_trainListY.append(sequence_Y)

    # Padding : make sure that each sample sequence has the same length (maxSeqLength)
    # X case
    # null_value = np.repeat(0, ARGS.numberOfInputCUIInts+ARGS.numberOfInputCCSInts) if ARGS.withCCS else np.repeat(0, ARGS.numberOfInputCUIInts)
    null_value = np.repeat(0, ARGS.numberOfInputCCSInts)
    for adlist in vectors_trainListX:
        for i in range(maxSeqLength - len(adlist)):
            adlist.append(null_value)
    # Y case
    null_value = np.repeat(0, 3)
    for adlist in mortality_trainListY:
        for i in range(maxSeqLength - len(adlist)):
            adlist.append(null_value)

    # Randomize in dimension 0 (patients order) keeping the notes and mortality in sync
    mapIndexPosition = list(zip(vectors_trainListX, mortality_trainListY))
    random.shuffle(mapIndexPosition)
    vectors_trainListX, mortality_trainListY = zip(*mapIndexPosition)

    return vectors_trainListX, mortality_trainListY


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def train():
    data_x, data_y = load_tensors()

    print("Available GPU :", torch.cuda.is_available())
    torch.cuda.set_device(0)

    sizedata = len(data_x)
    print("Data of size:", sizedata)
    # Split dataset into 5 sub-datasets
    splitted_x = list(split(data_x, 5))
    splitted_y = list(split(data_y, 5))
    k = ARGS.kFold

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

        with torch.cuda.device(0):
            # Hyperparameters :
            epochs = ARGS.nEpochs
            batchsize = ARGS.batchSize
            learning_rate = ARGS.lr
            log_interval = 2
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Data loader
            tensor_x = torch.Tensor(np.concatenate([np.array([np.array(plist, dtype=np.uint8) for plist in sublist]) for sublist in X_train]).tolist()).cuda()
            tensor_y = torch.Tensor(np.concatenate([np.array([np.array(plist, dtype=np.uint8) for plist in sublist]) for sublist in Y_train]).tolist()).cuda()
            print("Shape X:", tensor_x.size(), "Shape Y:", tensor_y.size())
            dataset = dt.TensorDataset(tensor_x, tensor_y)  # create your dataset
            print("TRAINLOADER")
            train_loader = dt.DataLoader(
                dataset,
                batch_size=batchsize,
                shuffle=True)
            
            # Training
            for epoch in range(epochs):
                h = model.init_hidden()
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = Variable(data.float()), Variable(target.float())
                    if data.size(0) != ARGS.batchSize:
                        continue
                    # cstate, hstate = h
                    # h = (cstate.detach(), hstate.detach())
                    h = h.detach()
                    optimizer.zero_grad()
                    net_out, h = model(data, h)
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
            h = model.init_hidden()
            for (data, targets) in test_loader:
                x, labels = Variable(data.float()), Variable(targets.float())
                if x.size(0) != ARGS.batchSize:
                    continue
                outputs, h = model(x, h)
                outputs = outputs.detach().cpu().numpy()
                x = x.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                # print("X Shape", np.shape(x), "Y Shape", np.shape(labels), "Predict Shape", np.shape(outputs))
                for batch_true, batch_prob in zip(labels, outputs):
                    for adm_true, adm_prob in zip(batch_true, batch_prob):
                        if torch.max(torch.from_numpy(adm_true)) != 1:
                            break
                        YTRUE = np.concatenate((YTRUE, [adm_true]), axis=0) if YTRUE is not None else [adm_true]
                        YPROBA = np.concatenate((YPROBA, [adm_prob]), axis=0) if YPROBA is not None else [adm_prob]
            ROC_avg_score=roc(YTRUE, YPROBA, average='micro', multi_class='ovr')
            print("ROC Average Score:", ROC_avg_score)
            AUC_folds.append(ROC_avg_score)

    # Output score of each fold + average
    print("AUC_scores: ", AUC_folds)
    print("Mean: ", sum(AUC_folds)/len(AUC_folds))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdata', type=str, default='prepared_data.npz', metavar='<visit_file>')
    parser.add_argument('--inputdata2', type=str, default='prepared_data_deathTime.npz', metavar='<visit_file>')
    parser.add_argument('--hiddenDimSize', type=int, default=200, help='Size of GRU hidden layer')
    parser.add_argument('--numLayers', type=int, default=1, help='Number of GRU layers')
    parser.add_argument('--batchSize', type=int, default=10, help='Batch size.')
    parser.add_argument('--nEpochs', type=int, default=1500, help='Number of training iterations.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--dropOut', type=float, default=0.5, help='GRU Dropout.')
    parser.add_argument('--kFold', type=int, default=5, help='K value (int) of K-fold cross-validation.')
    parser.add_argument('--withCCS', help='add CCS feature in input.')

    ARGStemp = parser.parse_args()
    return ARGStemp


if __name__ == '__main__':
    global ARGS
    ARGS = parse_arguments()
    train()
