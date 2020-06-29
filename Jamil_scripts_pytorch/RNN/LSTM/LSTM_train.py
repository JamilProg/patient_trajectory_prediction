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


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        ARGS.inputdim = ARGS.numberOfInputCUIInts
        self.num_classes = ARGS.numberOfInputCCSInts
        self.num_layers = ARGS.numLayers
        self.hidden_size = ARGS.hiddenDimSize                   
        
        self.lstm = nn.LSTM(input_size=ARGS.inputdim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
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
	#-------------------------------------------
	# diagnoses input (labels)
	# diagnoses_trainMapY = pickle.load(open(ARGS.inputFileDiagnoses, 'rb'))
	# print("-> " + str(len(diagnoses_trainMapY)) + " patients' diagnoses at dimension 0 for file: "+ ARGS.inputFileDiagnoses)
	# ARGS.numberOfOutputCodes = 270
    ARGS.numberOfOutputCodes = len(setOfDistinctCCSs)
    print('Remaining patients:', len(subjecttoadm_map))
	# Convert everything to list of list of list (patient x admission x CUInote_vector/diagnoses to ease the manipulation in batches
    vectors_trainListX = []
    diagnoses_trainListY = []
    # hadm_id_List = []
    maxSeqLength = 0
    for pID, adList in subjecttoadm_map.items():
        sequence_X = []
        sequence_Y = []
        if len(adList)-1 > maxSeqLength:
            maxSeqLength = len(adList)-1
        for i, adm in enumerate(adList):
            # hadm_id_List.append(adm[0])
            if i+1 == len(adList):
                # Avoid adding the last admission in X
                one_hot_CCS = [0] * ARGS.numberOfInputCCSInts
                for ccs_int in adm[3]:
                    one_hot_CCS[ccstoint[ccs_int]] = 1
                # diagnoses_trainListY.append(one_hot_CCS)
                sequence_Y.append(one_hot_CCS)
                continue
            one_hot_CUI = [0] * ARGS.numberOfInputCUIInts
            one_hot_CCS = [0] * ARGS.numberOfInputCCSInts
            for cui_int in adm[2]:
                one_hot_CUI[cuitoint[cui_int]] = 1
            for ccs_int in adm[3]:
                one_hot_CCS[ccstoint[ccs_int]] = 1
            # one_hot_X = one_hot_CUI + one_hot_CCS
            # vectors_trainListX.append(one_hot_X)
            sequence_X.append(one_hot_CUI)
            if i != 0:
                # Add every admission diagnoses in Y but the first one's diagnoses
                # diagnoses_trainListY.append(one_hot_CCS)
                sequence_Y.append(one_hot_CCS)
        vectors_trainListX.append(sequence_X)
        diagnoses_trainListY.append(sequence_Y)
    
    # Padding : make sure that each sample sequence has the same length (maxSeqLength)
    # X case
    null_value = np.repeat(0, ARGS.numberOfInputCUIInts)
    for adlist in vectors_trainListX:
        for i in range(maxSeqLength - len(adlist)):
            adlist.append(null_value)
    # Y case
    null_value = np.repeat(0, ARGS.numberOfInputCCSInts)
    for adlist in diagnoses_trainListY:
        for i in range(maxSeqLength - len(adlist)):
            adlist.append(null_value)

    # Randomize in dimension 0 (patients order) keeping the notes and diagnoses in sync
    # vectors_trainListX, diagnoses_trainListY, hadm_id_List = shuffle(notesVectors_trainListX, diagnoses_trainListY, hadm_id_List)
    mapIndexPosition = list(zip(vectors_trainListX, diagnoses_trainListY))
    random.shuffle(mapIndexPosition)
    vectors_trainListX, diagnoses_trainListY = zip(*mapIndexPosition)

    # Create train and test sets for each note
    sizedata = len(vectors_trainListX)
    vectors_testListX = vectors_trainListX[int(math.ceil(0.9*sizedata)):sizedata]
    vectors_trainListX = vectors_trainListX[0:int(math.ceil(0.9*sizedata))]

    # Create train and test sets for diagnoses
    diagnoses_testListY = diagnoses_trainListY[int(math.ceil(0.9*sizedata)):sizedata]
    diagnoses_trainListY = diagnoses_trainListY[0:int(math.ceil(0.9*sizedata))]
    # Save data for the test script, no need to save the train data
    pickle.dump(vectors_testListX, open('X-test.data', 'wb'), -1)
    pickle.dump(diagnoses_testListY, open('Y-test.data', 'wb'), -1)

    # Saving hadm_id of the test data, so that it becomes possible to find the original records
    # hadm_id_testList = hadm_id_List[int(math.ceil(0.9*numberOfPatients)):numberOfPatients]
    # hadm_file = open("HADM_ID-test.txt", "w")
    # hadm_file.write('output_index: subject_id, hadm_id'+'\n')
    # output_index = 0
    # for i in range(len(hadm_id_testList)):
    # 	for admission in hadm_id_testList[i][1]:
    # 		hadm_file.write(str(output_index)+': '+str(hadm_id_testList[i][0])+','+str(admission)+'\n')
    # 		output_index += 1
    # hadm_file.close()

    return vectors_trainListX, vectors_testListX, diagnoses_trainListY, diagnoses_testListY


def train():
    X_train, X_test, Y_train, Y_test = load_tensors()
    print("Available GPU :", torch.cuda.is_available())
    model = Network()
    epochs = ARGS.nEpochs
    batchsize = ARGS.batchSize
    learning_rate = ARGS.lr
    log_interval = 2
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Data loader
    X_train=np.array([np.array([np.array(unelist, dtype=np.float32) for unelist in xi]) for xi in X_train])
    Y_train=np.array([np.array([np.array(unelist, dtype=np.float32) for unelist in xi]) for xi in Y_train])
    tensor_x = torch.from_numpy(X_train) # transform to torch tensor
    print("X_dataset_shape=",tensor_x.shape)
    tensor_y = torch.from_numpy(Y_train)
    print("Y_dataset_shape=",tensor_y.shape)
    dataset = dt.TensorDataset(tensor_x, tensor_y) # create your dataset
    
    train_loader = dt.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True)

    # run the main training loop
    for epoch in range(epochs):
        h = model.init_hidden()
        for batch_idx, (data, target) in enumerate(train_loader):
            # h = tuple([Variable(e).data for e in h])
            data, target = Variable(data), Variable(target)
            if data.size(0) != ARGS.batchSize:
                # We have to ignore the last batch to avoid size errors
                continue
            #data, target = Variable(data).to(device), Variable(target).to(device)
            hstate, cstate = h
            h = (hstate.detach(), cstate.detach())
            optimizer.zero_grad()
            net_out, h = model(data, h)
            loss = criterion(net_out, target)
            # loss = criterion(net_out, torch.max(target,1)[1])
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: '.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader)))
                print(loss.data)
    # saving model
    torch.save(model.state_dict(), ARGS.outFile)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdata', type=str, default='prepared_data.npz', metavar='<visit_file>')
    parser.add_argument('--outFile', metavar='out_file', default='model_output.pt', help='Any file name to store the model.')
    # parser.add_argument('--maxConsecutiveNonImprovements', type=int, default=10, help='Training will run until reaching the maximum number of epochs without improvement before stopping the training')
    parser.add_argument('--hiddenDimSize', type=int, default=300, help='LSTM hidden layer size.')
    parser.add_argument('--batchSize', type=int, default=100, help='Batch size.')
    parser.add_argument('--nEpochs', type=int, default=600, help='Number of training iterations.')
    parser.add_argument('--numLayers', type=int, default=1, help='Number of LSTM layers.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    
    ARGStemp = parser.parse_args()
    return ARGStemp


if __name__ == '__main__':
    global ARGS
    ARGS = parse_arguments()
    train()
