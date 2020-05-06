#################################################################################################
# author: junio@usp.br - Jose F Rodrigues-Jr
# SCRIPT DL-TextBased-Prediction.py INPUT: file "01.preprocessed_notes.npz" and file "02.preprocessed_diagnoses_270.npz"
# (produced by the two earlier scripts)
# OUTPUT: -> printout of a sequence of pairs (epoch, cross-entropy error), that is, the script will perform the whole training (using
# all the data) a certain number of times; each time is called "epoch" in the usual terminology. After each epoch, it
# reports the errors of the predictor, which, in turn, are back-propagated throughout the network so that the predictor
# can learn from its mistakes.
# ->file "03.HADM_ID-test.txt". It connects the output of script "04.DL-CUIBased-Prediction-test.py"
# by an OUTPUT_INDEX, so that you can find the original record in the dataset my means of subject_id and hadm_id
# ->file "03.model_output.npz": this file carries the set of weights that define the model after the last epoch - together
# with the Deep Learning architecture, this is the predictor.
#OUTPUT MEANING: in the literature, it is usual to see plots epoch x error so that one can verify whether the predictor is
#learning. The file is saved so that one can perform predictions over unseen data, that is, data that the model hasn't ever
#seen.
#COMMAND LINE: "python DL-TextBased-Prediction.py" (provided all the files can be found and all the libraries are installed)
#################################################################################################
import random
import math
import cPickle as pickle
import os
from collections import OrderedDict
import argparse
import theano
import theano.tensor as T
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import sys
from sklearn.utils import shuffle

global ARGS
global tPARAMS

def unzip(zipped):
	new_params = OrderedDict()
	for key, value in zipped.iteritems():
		new_params[key] = value.get_value()
	return new_params

def numpy_floatX(data):
	return np.asarray(data, dtype=config.floatX)

def prepareTensors(train_listOflistOflistX, train_listOflistOflistY):
	nVisitsOfEachPatient_List = np.array([len(seq) for seq in train_listOflistOflistX]) - 1
	numberOfPatients = len(train_listOflistOflistX)
	maxNumberOfAdmissions = np.max(nVisitsOfEachPatient_List)

	x_notesVectors_tensorf = np.zeros((maxNumberOfAdmissions, numberOfPatients, ARGS.numberOfInputCUIInts)).astype(config.floatX)
	y_hotvectors_tensor = np.zeros((maxNumberOfAdmissions, numberOfPatients, ARGS.numberOfOutputCodes)).astype(config.floatX)
	mask = np.zeros((maxNumberOfAdmissions, numberOfPatients)).astype(config.floatX)

	for idx, train_patient_listOflist in enumerate(train_listOflistOflistX):
		for i_th_visit, vector_listOfCUIs in enumerate(train_patient_listOflist[:-1]): #ignores the last admission, which is not part of the training
			for ithCUI in vector_listOfCUIs:
				x_notesVectors_tensorf[i_th_visit, idx, ithCUI] = 1
	for idx, train_patient_listOflist in enumerate(train_listOflistOflistY):
		for i_th_visit, adm_listOfDiagCodes in enumerate(train_patient_listOflist[1:]):  #label_matrix[1:] = all but the first admission slice, not used to evaluate (this is the answer)
			for code in adm_listOfDiagCodes:
				y_hotvectors_tensor[i_th_visit, idx, code] = 1
		mask[:nVisitsOfEachPatient_List[idx], idx] = 1.

	nVisitsOfEachPatient_List = np.array(nVisitsOfEachPatient_List, dtype=config.floatX)
	return x_notesVectors_tensorf, y_hotvectors_tensor, mask, nVisitsOfEachPatient_List


#initialize model tPARAMS
def init_params_MinGRU(previousDimSize):
	for count, hiddenDimSize in enumerate(ARGS.hiddenDimSize):  #by default: 0, 200; 1, 200 according to enumerate
		#http://philipperemy.github.io/xavier-initialization/
		xavier_variance = math.sqrt(6.0/float(previousDimSize+hiddenDimSize))
		tPARAMS['fWf_'+str(count)] = theano.shared(np.random.normal(0., xavier_variance, (previousDimSize, hiddenDimSize)).astype(config.floatX), name='fWf_'+str(count))
		tPARAMS['fUf_' + str(count)] = theano.shared(np.identity(hiddenDimSize).astype(config.floatX), name='fUf_' + str(count))
		tPARAMS['fbf_'+str(count)] = theano.shared(np.zeros(hiddenDimSize).astype(config.floatX), name='fbf_'+str(count))

		tPARAMS['fWh_'+str(count)] = theano.shared(np.random.normal(0., xavier_variance, (previousDimSize, hiddenDimSize)).astype(config.floatX), name='fWh_'+str(count))
		tPARAMS['fUh_' + str(count)] = theano.shared(np.identity(hiddenDimSize).astype(config.floatX), name='fUh_' + str(count))
		tPARAMS['fbh_'+str(count)] = theano.shared(np.zeros(hiddenDimSize).astype(config.floatX), name='fbh_'+str(count))

		previousDimSize = hiddenDimSize
	tPARAMS['fResAlpha'] = theano.shared(0.1, name='fResAlpha')

	return previousDimSize


def fMinGRU_layer(inputTensor, layerIndex, hiddenDimSize, mask=None):
	# MinGRU: https://arxiv.org/pdf/1603.09420.pdf
	maxNumberOfVisits = inputTensor.shape[0]
	batchSize = inputTensor.shape[1]

	Wf = T.dot(inputTensor,tPARAMS['fWf_' + layerIndex])
	Wh = T.dot(inputTensor,tPARAMS['fWh_' + layerIndex])

	def stepFn(stepMask, wf, wh, h_previous, h_previous_previous):
		f = T.nnet.sigmoid(wf + T.dot(h_previous,tPARAMS['fUf_' + layerIndex]) + tPARAMS['fbf_' + layerIndex])
		h_intermediate = tPARAMS['fResAlpha']*h_previous_previous + T.tanh(wh + T.dot(f * h_previous, tPARAMS['fUh_' + layerIndex]) + tPARAMS['fbh_' + layerIndex])
		h_new = ((1. - f) * h_previous) + f * h_intermediate
		h_new = stepMask[:, None] * h_new + (1. - stepMask)[:,None] * h_previous
		return h_new, h_previous # becomes h_previous in the next iteration

	#here, we unfold the RNN
	results, _ = theano.scan(fn=stepFn,  # function to execute
							 sequences=[mask, Wf, Wh],  # input to stepFn
							 outputs_info=[T.alloc(numpy_floatX(0.0), batchSize, hiddenDimSize),T.alloc(numpy_floatX(0.0), batchSize, hiddenDimSize)], #initial h_previous
							 name='fMinGRU_layer' + layerIndex,  # labeling for debug
							 n_steps=maxNumberOfVisits)  # number of times to execute (d0 times, once for each time step)

	return results[0]

def init_params_output_layer(previousDimSize):
	xavier_variance = math.sqrt(2.0 / float(previousDimSize + ARGS.numberOfOutputCodes))
	tPARAMS['W_output'] = theano.shared(np.random.normal(0., xavier_variance, (previousDimSize, ARGS.numberOfOutputCodes)).astype(config.floatX), 'W_output')
	tPARAMS['b_output'] = theano.shared(np.zeros(ARGS.numberOfOutputCodes).astype(config.floatX), name='b_output')
	tPARAMS['olrelu'] = theano.shared(0.1, name='olrelu')

def dropout(nDimensionalData):
	randomS = RandomStreams(13713)
	newTensor = nDimensionalData * randomS.binomial(nDimensionalData.shape, p=ARGS.dropoutRate, dtype=nDimensionalData.dtype)
	#https://www.quora.com/How-do-you-implement-a-dropout-in-deep-neural-networks
	return newTensor

def build_model():
	xf = T.tensor3('xf', dtype=config.floatX)
	y = T.tensor3('y', dtype=config.floatX)
	mask = T.matrix('mask', dtype=config.floatX)

	nVisitsOfEachPatient_List = T.vector('nVisitsOfEachPatient_List', dtype=config.floatX)
	nOfPatients = nVisitsOfEachPatient_List.shape[0]
	maxNumberOfAdmissions = xf.shape[0]

	flowing_tensorf = xf

	for i, hiddenDimSize in enumerate(ARGS.hiddenDimSize):
		flowing_tensorf = fMinGRU_layer(flowing_tensorf, str(i), hiddenDimSize, mask=mask)
		flowing_tensorf = dropout(flowing_tensorf)

	results, _ = theano.scan(
		lambda theFlowingTensor: T.nnet.softmax(T.nnet.relu(T.dot(theFlowingTensor, tPARAMS['W_output']) + tPARAMS['b_output'], tPARAMS['olrelu'])),
		sequences=[flowing_tensorf],
		outputs_info=None,
		name='softmax_layer',
		n_steps=maxNumberOfAdmissions)

	flowing_tensor = results * mask[:, :, None]
	epislon = 1e-8

	cross_entropy = -(y * T.log(flowing_tensor + epislon) + (1. - y) * T.log(1. - flowing_tensor + epislon))
	# the complete crossentropy equation is -1/n* sum(cross_entropy); where n is the number of elements
	# see http://neuralnetworksanddeeplearning.com/chap3.html#regularization
	prediction_loss = cross_entropy.sum(axis=2).sum(axis=0) / nVisitsOfEachPatient_List

	L2_regularized_loss = T.mean(prediction_loss) + ARGS.LregularizationAlpha*(tPARAMS['W_output'] ** 2).sum()
	MODEL = L2_regularized_loss
	return xf, y, mask, nVisitsOfEachPatient_List, MODEL


#this code comes originally from deeplearning.net/tutorial/LSTM.html
#http://ruder.io/optimizing-gradient-descent/index.html#adadelta
#https://arxiv.org/abs/1212.5701
def addAdadeltaGradientDescent(grads, xf, y, mask, nVisitsOfEachPatient_List, MODEL):
	zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in tPARAMS.iteritems()]
	running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % k) for k, p in tPARAMS.iteritems()]
	running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k) for k, p in tPARAMS.iteritems()]

	zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
	rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

	TRAIN_MODEL_COMPILED = theano.function([xf, y, mask, nVisitsOfEachPatient_List], MODEL, updates=zgup + rg2up, name='adadelta_TRAIN_MODEL_COMPILED')

	updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
	ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
	param_up = [(p, p + ud) for p, ud in zip(tPARAMS.values(), updir)]

	UPDATE_WEIGHTS_COMPILED = theano.function([], [], updates=ru2up + param_up, name='adadelta_UPDATE_WEIGHTS_COMPILED')
	return TRAIN_MODEL_COMPILED, UPDATE_WEIGHTS_COMPILED


def load_data():
	#-----------------------------------------
	#pickle input map - each entry is a pair (subject_id, [(hadm_id,admittime, [CUIsvector])]
	notesVectors_trainMapX = pickle.load(open(ARGS.inputFileNotes, 'rb'))
	setOfDistinctCUIs = set()
	for subject in notesVectors_trainMapX:
		patientData = notesVectors_trainMapX[subject]
		for ithAdmis in patientData:
			for CUIcode in ithAdmis[2]:
				setOfDistinctCUIs.add(CUIcode)
	print("-> " + str(len(notesVectors_trainMapX)) + " patients' CUI notes at dimension 0 for file: "+ ARGS.inputFileNotes)
	ARGS.numberOfInputCUIInts = len(setOfDistinctCUIs)
	print('Number of distinct CUI codes: ' + str(ARGS.numberOfInputCUIInts))
	file = open('NumberOfCUIcodes.txt', "a")
	file.write(str(ARGS.numberOfInputCUIInts))
	file.close()
	#-------------------------------------------
	# diagnoses input (labels)
	diagnoses_trainMapY = pickle.load(open(ARGS.inputFileDiagnoses, 'rb'))
	print("-> " + str(len(diagnoses_trainMapY)) + " patients' diagnoses at dimension 0 for file: "+ ARGS.inputFileDiagnoses)
	ARGS.numberOfOutputCodes = 270

	#make sure every note admission matches a diagnoses admission
	number_of_mismatching_records = 0

	#toss off the set difference
	for subject in diagnoses_trainMapY.keys():
		if subject not in notesVectors_trainMapX.keys():
			number_of_mismatching_records+=1
			del diagnoses_trainMapY[subject]

	for subject in notesVectors_trainMapX.keys():
		if subject not in diagnoses_trainMapY.keys():
			number_of_mismatching_records += 1
			del notesVectors_trainMapX[subject]

	#toss off mismatching records
	for subject, notes_admissions in notesVectors_trainMapX.iteritems():
		diagnoses_admissions = diagnoses_trainMapY[subject]
		if len(notes_admissions) != len(diagnoses_admissions):
			# exclude patient with number of notes admissions different of the number of diagnoses admissions
			del diagnoses_trainMapY[subject]
			number_of_mismatching_records += 1
			continue

		for i in range(len(notes_admissions)):
			# exclude patient with a note hadm_id not being the same as the corresponding diagnoses hadm_id (in temporal order)
			if notes_admissions[i][0] != diagnoses_admissions[i][0]:
				del diagnoses_trainMapY[subject]
	# toss off the set difference again, as the data changed
	for subject in notesVectors_trainMapX.keys():
		if subject not in diagnoses_trainMapY.keys():
			number_of_mismatching_records += 1
			del notesVectors_trainMapX[subject]
			
	print('Number of mismatches between notes and diagnoses admissions: ' + str(number_of_mismatching_records))
	print('Remaining patients: ' + str(len(notesVectors_trainMapX)) + '  '+  str(len(diagnoses_trainMapY)))

	#convert everything to list of list of list (patient x admission x CUInote_vector/diagnoses to ease the manipulation in batches
	notesVectors_trainListX = []
	diagnoses_trainListY = []
	hadm_id_List = []
	for subject, notes_admissions in notesVectors_trainMapX.iteritems():
		diagnoses_admissions = diagnoses_trainMapY[subject]
		notesVectors_subjectList = []
		diagnoses_subjectList = []
		hadm_id_subjectList = []
		for ithAdmission in range(len(notes_admissions)):
			note_admission = notes_admissions[ithAdmission]
			diagnose_admission = diagnoses_admissions[ithAdmission]
			# asserting only, as we already verified everything; diagnose_admission[0] = hadm_id
			if note_admission[0] != diagnose_admission[0]:
				print 'ERROR: divergent hadm_id for note and diagnoses for the same patient'
				sys.exit(0)
			notesVectors_subjectList.append(note_admission[2])
			diagnoses_subjectList.append(diagnose_admission[2])
			hadm_id_subjectList.append(diagnose_admission[0])
		notesVectors_trainListX.append(notesVectors_subjectList)
		diagnoses_trainListY.append(diagnoses_subjectList)
		hadm_id_List.append((subject,hadm_id_subjectList))

	#randomize in dimension 0 (patients order) keeping the notes and diagnoses in sync
	notesVectors_trainListX, diagnoses_trainListY, hadm_id_List = shuffle(notesVectors_trainListX, diagnoses_trainListY,hadm_id_List)
	numberOfPatients = len(notesVectors_trainListX)
	#create train and test sets for notes
	notesVectors_testListX = notesVectors_trainListX[int(math.ceil(0.9*numberOfPatients)):numberOfPatients]
	notesVectors_trainListX = notesVectors_trainListX[0:int(math.ceil(0.9*numberOfPatients))]
	#save data for the test script, no need to save the train data
	pickle.dump(notesVectors_testListX, open(ARGS.inputFileNotes + '-test', 'wb'), -1)

	# create train and test sets for diagnoses
	diagnoses_testListY = diagnoses_trainListY[int(math.ceil(0.9*numberOfPatients)):numberOfPatients]
	diagnoses_trainListY = diagnoses_trainListY[0:int(math.ceil(0.9*numberOfPatients))]
	#save data for the test script, no need to save the train data
	pickle.dump(diagnoses_testListY, open(ARGS.inputFileDiagnoses + '-test', 'wb'), -1)

	#saving hadm_id of the test data, so that it becomes possible to find the original records
	hadm_id_testList = hadm_id_List[int(math.ceil(0.9*numberOfPatients)):numberOfPatients]
	hadm_file = open("03.HADM_ID-test.txt", "w")
	hadm_file.write('output_index: subject_id, hadm_id'+'\n')
	output_index = 0
	for i in range(len(hadm_id_testList)):
		for admission in hadm_id_testList[i][1]:
			hadm_file.write(str(output_index)+': '+str(hadm_id_testList[i][0])+','+str(admission)+'\n')
			output_index += 1
	hadm_file.close()

	return notesVectors_trainListX, notesVectors_testListX, diagnoses_trainListY, diagnoses_testListY

#the performance computation uses the test data and returns the cross entropy measure
def performEvaluation(TEST_MODEL_COMPILED, test_SetX, test_SetY):
	batchSize = ARGS.batchSize

	n_batches = int(np.ceil(float(len(test_SetX[0])) / float(batchSize))) #default batch size is 100
	crossEntropySum = 0.0
	dataCount = 0.0
	#computes de crossEntropy for all the elements in the test_Set, using the batch scheme of partitioning
	for index in xrange(n_batches):
		batchX = test_SetX[index * batchSize:(index + 1) * batchSize]
		batchY = test_SetY[index * batchSize:(index + 1) * batchSize]
		xf, y, mask, nVisitsOfEachPatient_List = prepareTensors(batchX,batchY)
		crossEntropy = TEST_MODEL_COMPILED(xf, y, mask, nVisitsOfEachPatient_List)

		#accumulation by simple summation taking the batch size into account
		crossEntropySum += crossEntropy * len(batchX)
		dataCount += float(len(batchX))
		#At the end, it returns the mean cross entropy considering all the batches
	return n_batches, crossEntropySum / dataCount

def train_model():
	print '==> data loading'
	notesVectors_trainListX, notesVectors_testListX, diagnoses_trainListY, diagnoses_testListY = load_data()
	prepareTensors(notesVectors_trainListX, diagnoses_trainListY)
	previousDimSize = ARGS.numberOfInputCUIInts

	print '==> parameters initialization'
	print('Using neuron type Minimal Gated Recurrent Unit')
	previousDimSize = init_params_MinGRU(previousDimSize)
	init_params_output_layer(previousDimSize)

	print '==> model building'
	xf, y, mask, nVisitsOfEachPatient_List, MODEL = build_model()
	grads = T.grad(theano.gradient.grad_clip(MODEL, -0.3, 0.3), wrt=tPARAMS.values())
	TRAIN_MODEL_COMPILED, UPDATE_WEIGHTS_COMPILED = addAdadeltaGradientDescent(grads, xf, y, mask, nVisitsOfEachPatient_List, MODEL)

	print '==> training and validation'
	batchSize = ARGS.batchSize
	n_batches = int(np.ceil(float(len(notesVectors_trainListX)) / float(batchSize)))
	TEST_MODEL_COMPILED = theano.function(inputs=[xf, y, mask, nVisitsOfEachPatient_List], outputs=MODEL, name='TEST_MODEL_COMPILED')

	bestValidationCrossEntropy = 1e20
	bestValidationEpoch = 0
	bestModelFileName = ''

	iImprovementEpochs = 0
	iConsecutiveNonImprovements = 0
	epoch_counter = 0
	for epoch_counter in xrange(ARGS.nEpochs):
		iteration = 0
		trainCrossEntropyVector = []
		for index in random.sample(range(n_batches), n_batches):
			batchX, batchY = notesVectors_trainListX[index*batchSize:(index+1)*batchSize], diagnoses_trainListY[index*batchSize:(index+1)*batchSize]
			xf, y, mask, nVisitsOfEachPatient_List = prepareTensors(batchX,batchY)
			xf += np.random.normal(0, 0.1, xf.shape)  #add gaussian noise as a means to reduce overfitting

			trainCrossEntropy = TRAIN_MODEL_COMPILED(xf, y, mask, nVisitsOfEachPatient_List)
			trainCrossEntropyVector.append(trainCrossEntropy)
			UPDATE_WEIGHTS_COMPILED()
			iteration += 1
		#----------test -> uses TEST_MODEL_COMPILED
		#evaluates the network with the notesVectors_testMapX
		print('-> Epoch: %d, mean cross entropy considering %d TRAINING batches: %f' % (epoch_counter, n_batches, np.mean(trainCrossEntropyVector)))
		nValidBatches, validationCrossEntropy = performEvaluation(TEST_MODEL_COMPILED, notesVectors_testListX, diagnoses_testListY)
		print('			 mean cross entropy considering %d VALIDATION batches: %f' % (nValidBatches, validationCrossEntropy))
		if validationCrossEntropy < bestValidationCrossEntropy:
			iImprovementEpochs += 1
			iConsecutiveNonImprovements = 0
			bestValidationCrossEntropy = validationCrossEntropy
			bestValidationEpoch = epoch_counter

			tempParams = unzip(tPARAMS)
			bestModelFileName = ARGS.outFile
			if os.path.exists(bestModelFileName):
				os.remove(bestModelFileName)
			np.savez_compressed(bestModelFileName, **tempParams)
		else:
			print('Epoch ended without improvement.')
			iConsecutiveNonImprovements += 1
		if iConsecutiveNonImprovements > ARGS.maxConsecutiveNonImprovements: #default is 10
			break
	#Best results
	print('--------------SUMMARY--------------')
	print('The best VALIDATION cross entropy occurred at epoch %d, the value was of %f ' % (bestValidationEpoch, bestValidationCrossEntropy))
	print('Best model file: ' + bestModelFileName)
	print('Number of improvement epochs: ' + str(iImprovementEpochs) + ' out of ' + str(epoch_counter+1) + ' possible improvements.')
	print('Note: the smaller the cross entropy, the better.')
	print('-----------------------------------')

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputFileNotes', type=str, default='01.preprocessed_notes.npz', metavar='<visit_file>')
	parser.add_argument('--inputFileDiagnoses', type=str,  default='02.preprocessed_diagnoses_270.npz', metavar='<visit_file>')
	parser.add_argument('--outFile', metavar='out_file', default='03.model_output.npz', help='Any file name to store the model.')
	parser.add_argument('--maxConsecutiveNonImprovements', type=int, default=10, help='Training will run until reaching the maximum number of epochs without improvement before stopping the training')
	parser.add_argument('--hiddenDimSize', type=str, default='[270]', help='Number of layers and their size - for example [100,200] refers to two layers with 100 and 200 nodes.')
	parser.add_argument('--batchSize', type=int, default=100, help='Batch size.')
	parser.add_argument('--nEpochs', type=int, default=1000, help='Number of training iterations.')
	parser.add_argument('--LregularizationAlpha', type=float, default=0.001, help='Alpha regularization for L2 normalization')
	parser.add_argument('--dropoutRate', type=float, default=0.5, help='Dropout probability.')

	ARGStemp = parser.parse_args()
	hiddenDimSize = [int(strDim) for strDim in ARGStemp.hiddenDimSize[1:-1].split(',')]
	ARGStemp.hiddenDimSize = hiddenDimSize
	return ARGStemp



if __name__ == '__main__':
	#os.environ["MKL_THREADING_LAYER"] = "GNU"
	global tPARAMS
	tPARAMS = OrderedDict()
	global ARGS
	ARGS = parse_arguments()

	train_model()
