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

        # Inputs to hidden layer linear transformation
        ARGS.inputdim = ARGS.numberOfInputCUIInts + ARGS.numberOfInputCCSInts
        # ARGS.inputdim = ARGS.numberOfInputCUIInts
        self.hidden = nn.Linear(ARGS.inputdim, ARGS.hiddenDimSize)
        # Hidden layer
        self.hidden2 = nn.Linear(ARGS.hiddenDimSize, ARGS.numberOfOutputCodes)

        # Define sigmoid activation and softmax output
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
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
    for pID, adList in subjecttoadm_map.items():
        for i, adm in enumerate(adList):
            # hadm_id_List.append(adm[0])
            if i+1 == len(adList):
                # Avoid adding the last admission in X
                one_hot_CCS = [0] * ARGS.numberOfInputCCSInts
                for ccs_int in adm[3]:
                    one_hot_CCS[ccstoint[ccs_int]] = 1
                diagnoses_trainListY.append(one_hot_CCS)
                continue
            one_hot_CUI = [0] * ARGS.numberOfInputCUIInts
            one_hot_CCS = [0] * ARGS.numberOfInputCCSInts
            for cui_int in adm[2]:
                one_hot_CUI[cuitoint[cui_int]] = 1
            for ccs_int in adm[3]:
                one_hot_CCS[ccstoint[ccs_int]] = 1
            # one_hot_X = one_hot_CUI + one_hot_CCS
            one_hot_X = one_hot_CUI
            vectors_trainListX.append(one_hot_X)
            if i != 0:
                # Add every admission diagnoses in Y but the first one's diagnoses
                diagnoses_trainListY.append(one_hot_CCS)

    # Randomize in dimension 0 (patients order) keeping the notes and diagnoses in sync
    # vectors_trainListX, diagnoses_trainListY, hadm_id_List = shuffle(notesVectors_trainListX, diagnoses_trainListY, hadm_id_List)
    mapIndexPosition = list(zip(vectors_trainListX, diagnoses_trainListY))
    random.shuffle(mapIndexPosition)
    vectors_trainListX, diagnoses_trainListY = zip(*mapIndexPosition)

    # Create train and test sets for notes
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Hyperparameters :
    epochs = ARGS.nEpochs
    batchsize = ARGS.batchSize
    learning_rate = ARGS.lr
    log_interval = 2
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Data loader
    tensor_x = torch.Tensor(np.array(X_train)) # transform to torch tensor
    print("X_dataset_shape=",tensor_x.shape)
    tensor_y = torch.Tensor(np.array(Y_train))
    print("Y_dataset_shape=",tensor_y.shape)
    dataset = dt.TensorDataset(tensor_x, tensor_y) # create your dataset
    train_size = int(len(dataset))
    print("train_size =", train_size)

    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = dt.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True)

    # run the main training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            #data, target = Variable(data), Variable(target)
            data, target = Variable(data).to(device), Variable(target).to(device)
            optimizer.zero_grad()
            net_out = model(data)
            loss = criterion(net_out, target)
            # loss = criterion(net_out, torch.max(target,1)[1])
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.data))
    # saving model
    torch.save(model.state_dict(), ARGS.outFile)




def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdata', type=str, default='prepared_data.npz', metavar='<visit_file>')
    parser.add_argument('--outFile', metavar='out_file', default='model_output.pt', help='Any file name to store the model.')
    # parser.add_argument('--maxConsecutiveNonImprovements', type=int, default=10, help='Training will run until reaching the maximum number of epochs without improvement before stopping the training')
    parser.add_argument('--hiddenDimSize', type=int, default=10000, help='Number of neurons in the hidden layer.')
    parser.add_argument('--batchSize', type=int, default=100, help='Batch size.')
    parser.add_argument('--nEpochs', type=int, default=5000, help='Number of training iterations.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    
    ARGStemp = parser.parse_args()
    return ARGStemp


if __name__ == '__main__':
    global ARGS
    ARGS = parse_arguments()
    train()
