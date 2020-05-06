#################################################################################################
# author: junio@usp.br - Jose F Rodrigues Jr
# INPUT: files"01.preprocessed_notes.npz-test" and "02.preprocessed_diagnoses_270.npz-test";
# and the model saved in file "03.model_output.npz"
# then, it performs predictions for the set of patients in the test files
# OUTPUT: multiple performance metrics used in Machine Learning
# (Precision@, Recall@, F1-Measure, and AUROC), and a readable output
# concerning what was predicted to each patient of the test set.
#################################################################################################
import numpy as np
import cPickle as pickle
from collections import OrderedDict
import argparse
import theano
import theano.tensor as T
from theano import config
from sklearn import metrics
global ARGS
global tPARAMS
import sys

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

	return x_notesVectors_tensorf, y_hotvectors_tensor, mask, nVisitsOfEachPatient_List


def loadModel():
	model = np.load(ARGS.modelFile)
	tPARAMS = OrderedDict()
	for key, value in model.iteritems():
		tPARAMS[key] = theano.shared(value, name=key)
	ARGS.numberOfInputCodes = model['fWf_0'].shape[0]
	return tPARAMS

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


def build_model():
	xf = T.tensor3('xf', dtype=config.floatX)
	mask = T.matrix('mask', dtype=config.floatX)
	maxNumberOfAdmissions = xf.shape[0]

	flowing_tensorf = xf

	for i, hiddenDimSize in enumerate(ARGS.hiddenDimSize):
		flowing_tensorf = fMinGRU_layer(flowing_tensorf, str(i), hiddenDimSize, mask=mask)

	results, _ = theano.scan(
		lambda theFlowingTensor: T.nnet.softmax(T.nnet.relu(T.dot(theFlowingTensor, tPARAMS['W_output']) + tPARAMS['b_output'], tPARAMS['olrelu'])),
		sequences=[flowing_tensorf],
		outputs_info=None,
		name='softmax_layer',
		n_steps=maxNumberOfAdmissions)

	MODEL = results * mask[:, :, None]
	return xf, mask, MODEL


def load_data():
	testSet_x = np.array(pickle.load(open(ARGS.inputFileNotes_test+'.test', 'rb')))

	def len_argsort(seq):
		return sorted(range(len(seq)), key=lambda x: len(seq[x]))

	sorted_index = len_argsort(testSet_x)
	testSet_x = [testSet_x[i] for i in sorted_index]

	return testSet_x


def load_data():
	# -----------------------------------------
	# textual vector input
	notesVectors_testListX = pickle.load(open(ARGS.inputFileNotes_test, 'rb'))
	print("-> " + str(len(notesVectors_testListX)) + " patients' notes at dimension 0 for file: " + ARGS.inputFileNotes_test)
	file = open('NumberOfCUIcodes.txt', "r")
	ARGS.numberOfInputCUIInts = int(file.readline())

	# -------------------------------------------
	# diagnoses input (labels)
	diagnoses_testListY = pickle.load(open(ARGS.inputFileDiagnoses_test, 'rb'))
	print("-> " + str(len(diagnoses_testListY)) + " patients' diagnoses at dimension 0 for file: " + ARGS.inputFileDiagnoses_test)
	ARGS.numberOfOutputCodes = 270
	file.close()

	# ALL THE FILES WERE RANDOMIZED PREVIOUSLY, DO NOT RANDOMIZE THEM HERE BECAUSE IT WOULD MISS THE SYNC TO FILE HADM_ID-test.txt
	return notesVectors_testListX, diagnoses_testListY


def testModel():
	print '==> model loading'
	global tPARAMS
	tPARAMS = loadModel()

	print '==> data loading'
	testSetX, testSetY = load_data()

	print '==> model rebuilding'
	xf, mask, MODEL = build_model()
	PREDICTOR_COMPILED = theano.function(inputs=[xf, mask], outputs=MODEL, name='PREDICTOR_COMPILED')

	print '==> model execution'
	nBatches = int(np.ceil(float(len(testSetX)) / float(ARGS.batchSize)))
	predictedY_list = []
	predictedProbabilities_list = []
	actualY_list = []

	#Execute once for each batch
	output_index = 0
	for batchIndex in range(nBatches):
		batchX = testSetX[batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
		batchY = testSetY[batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
		xf, y, mask, nVisitsOfEachPatient_List = prepareTensors(batchX, batchY)
		#retrieve the maximum number of admissions considering all the patients
		maxNumberOfAdmissions = np.max(nVisitsOfEachPatient_List)
		#make prediction
		predicted_y = PREDICTOR_COMPILED(xf, mask)

		#traverse the predicted results, once for each patient in the batch
		for ith_patient in range(predicted_y.shape[1]):
			predictedPatientSlice = predicted_y[:, ith_patient, :] #for patients with one single predicted admission - it is a vector
			#retrieve actual y from batch tensor -> actual codes, not the hotvector
			actual_y = batchY[ith_patient][1:]  #discard the first admission, which could not be predicted

			#retrieves the last admission
			#ith_admission = nVisitsOfEachPatient_List[ith_patient]-1
			for ith_admission in range(nVisitsOfEachPatient_List[ith_patient]): #here I know the number of (predicted) admissions of the ith-patient
				#convert array of actual answers to list
				actualY_list.append(actual_y[ith_admission])
				#retrieves ith-admission prediction of the ith-patient
				ithPrediction = predictedPatientSlice[ith_admission]
				#since ithPrediction is a vector of probabilties with the same dimensionality of the hotvectors
				#enumerate is enough to retrieve the original codes (not the original CCS, the original after the preprocess script)
				enumeratedPrediction = [codeProbability_pair for codeProbability_pair in enumerate(ithPrediction)]
				#sort everything
				sortedPredictionsAll = sorted(enumeratedPrediction, key=lambda x: x[1],reverse=True)
				#creates trimmed list up to max(maxNumberOfAdmissions,30) elements
				sortedTopPredictions = sortedPredictionsAll[0:max(maxNumberOfAdmissions,30)]
				#here we simply toss off the probability and keep only the sorted codes
				sortedTopPredictions_indexes = [codeProbability_pair[0] for codeProbability_pair in sortedTopPredictions]
				#stores results in a list of lists - after processing all batches, predictedY_list stores all the prediction results
				predictedY_list.append(sortedTopPredictions_indexes)
				predictedProbabilities_list.append(sortedPredictionsAll)

				interpretOutPut = True #works for mimic only - either CCS or ICD-9 (must run preprocess before everything)
				if interpretOutPut:
					internalCodeToTextualDescriptionMAP = pickle.load(open('internalCodeToTextualDescriptionMAP.pickle', 'rb'))
					if ith_admission == 0:
						print '-----------------------------------------------'
						print '-----------------------------------------------'
						print 'PATIENT' + str(ith_patient) + ' with ' + str(nVisitsOfEachPatient_List[ith_patient] + 1) + ' original admissions'
						print 'Patient history'
						ZerothAdmission = batchY[ith_patient][0]
						print 'Admission....: ' + str(ith_admission) + ' => ' + str(ZerothAdmission)  # using for debug, I do not discard the first
						print '  Textual....: ' + str([(code, internalCodeToTextualDescriptionMAP[code]) for code in ZerothAdmission])
						print 'OUTPUT_INDEX: ' + str(output_index)
						output_index += 1
					print ''

					actualIthAdmission = actual_y[ith_admission]  # actual_y starts at actual admission 1
					sortedActualAdmis = sorted(actualIthAdmission, key=lambda x: x)
					print 'Admission....: ' + str(ith_admission + 1) + ' => ' + str(sortedActualAdmis)  # using for debug, I do not discard the first
					print '  Textual....: ' + str([(code,internalCodeToTextualDescriptionMAP[code]) for code in sortedActualAdmis])
					print 'OUTPUT_INDEX: ' + str(output_index)
					output_index += 1

					print 'Prediction...: ' + str(ith_admission + 1) + ' => ' + str(sortedTopPredictions_indexes)
					print '  Textual....: ' + str([(code, internalCodeToTextualDescriptionMAP[code]) for code in sortedTopPredictions_indexes[0:10]])

					intersection = set(sortedTopPredictions_indexes) & set(sortedActualAdmis)
					print '-Intersection: ' + str(sorted(intersection, key=lambda x: x))
					print '  Textual....: ' + str([(code, internalCodeToTextualDescriptionMAP[code]) for code in intersection])

					differenceToPreviousAdmis = set(intersection) - set(batchY[ith_patient][ith_admission]) #here, we have to use batch, because actual_y estarts at index 1
					print '-Difference..: ' + str(sorted(differenceToPreviousAdmis, key=lambda x: x)) + ' => predicted, but not in the previous admission; means robustness of the predictor'
					print '  Textual....: ' + str([(code, internalCodeToTextualDescriptionMAP[code]) for code in differenceToPreviousAdmis])
	#---------------------------------Report results using k=[10,20,30]
	print '==> computation of prediction results with constant k'
	recall_sum = [0.0, 0.0, 0.0]

	k_list = [10,20,30]
	for ith_admission in range(len(predictedY_list)):
		ithActualYSet = set(actualY_list[ith_admission])
		for ithK, k in enumerate(k_list):
			ithPredictedY = set(predictedY_list[ith_admission][:k])
			intersection_set = ithActualYSet.intersection(ithPredictedY)
			recall_sum[ithK] += len(intersection_set) / float(len(ithActualYSet)) # this is recall because the numerator is len(ithActualYSet)

	precision_sum = [0.0, 0.0, 0.0]
	k_listForPrecision = [1,2,3]
	for ith_admission in range(len(predictedY_list)):
		ithActualYSet = set(actualY_list[ith_admission])
		for ithK, k in enumerate(k_listForPrecision):
			ithPredictedY = set(predictedY_list[ith_admission][:k])
			intersection_set = ithActualYSet.intersection(ithPredictedY)
			precision_sum[ithK] += len(intersection_set) / float(k) # this is precision because the numerator is k \in [10,20,30]

	finalRecalls = []
	finalPrecisions = []
	for ithK, k in enumerate(k_list):
		finalRecalls.append(recall_sum[ithK] / float(len(predictedY_list)))
		finalPrecisions.append(precision_sum[ithK] / float(len(predictedY_list)))

	print 'Results for Recall@' + str(k_list)
	print str(finalRecalls[0])
	print str(finalRecalls[1])
	print str(finalRecalls[2])

	print 'Results for Precision@' + str(k_listForPrecision)
	print str(finalPrecisions[0])
	print str(finalPrecisions[1])
	print str(finalPrecisions[2])

	#---------------------------------Report results using k=lenght of actual answer vector
	print '==> computation of prediction results with dynamic k=lenght of actual answer vector times [1,2,3]'
	recall_sum = [0.0, 0.0, 0.0]
	precision_sum = [0.0, 0.0, 0.0]
	multiples_list = [0, 1, 2]
	for ith_admission in range(len(predictedY_list)):
		ithActualYSet = set(actualY_list[ith_admission])
		#print '--->Admission: ' + str(ith_admission)
		for m in multiples_list:
			k = len(ithActualYSet) * (m+1)
			#print 'K: ' + str(k)
			ithPredictedY = set(predictedY_list[ith_admission][:k])
			#print 'Prediction: ' + str(ithPredictedY)
			#print 'Actual: ' + str(ithActualYSet)
			intersection_set = ithActualYSet.intersection(ithPredictedY)
			#print 'Intersection: ' + str(intersection_set)
			recall_sum[m] += len(intersection_set) / float(len(ithActualYSet))
			precision_sum[m] += len(intersection_set) / float(k) # this is precision because the numerator is ithK \in [10,20,30]

	bReportDynamic_K = False
	if bReportDynamic_K:
		finalRecalls = []
		finalPrecisions = []
		for m in multiples_list:
			finalRecalls.append(recall_sum[m] / float(len(predictedY_list)))
			finalPrecisions.append(precision_sum[m] / float(len(predictedY_list)))

		print 'Results for Recall@k*1, Recall@k*2, and Recall@k*3'
		print str(finalRecalls[0])
		print str(finalRecalls[1])
		print str(finalRecalls[2])

		print 'Results for Precision@k*1, Precision@k*2, and Precision@k*3'
		print str(finalPrecisions[0])
		print str(finalPrecisions[1])
		print str(finalPrecisions[2])

	# ---------------------------------Write data for AUC-ROC computation
	fullListOfTrueYOutcomeForAUCROCAndPR_list = []
	fullListOfPredictedYProbsForAUCROC_list = []
	fullListOfPredictedYForPrecisionRecall_list = []
	for ith_admission in range(len(predictedY_list)):
		ithActualY = actualY_list[ith_admission]
		nActualCodes = len(ithActualY)
		ithPredictedProbabilities = predictedProbabilities_list[ith_admission]#[0:nActualCodes]
		ithPrediction = 0
		for predicted_code, predicted_prob in ithPredictedProbabilities:
			fullListOfPredictedYProbsForAUCROC_list.append(predicted_prob)
			#for precision-recall purposes, the nActual first codes correspond to what was estimated as correct answers
			if ithPrediction < nActualCodes:
				fullListOfPredictedYForPrecisionRecall_list.append(1)
			else:
				fullListOfPredictedYForPrecisionRecall_list.append(0)

			#the list fullListOfTrueYOutcomeForAUCROCAndPR_list corresponds to the true answer, either positive or negative
			#it is used for both Precision Recall and for AUCROC
			if predicted_code in ithActualY:
				fullListOfTrueYOutcomeForAUCROCAndPR_list.append(1)
				#file.write("1 " + str(predicted_prob) + '\n')
			else:
				fullListOfTrueYOutcomeForAUCROCAndPR_list.append(0)
				#file.write("0 " + str(predicted_prob) + '\n')
			ithPrediction += 1
	#file.close()

	#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
	print "Weighted AUC-ROC score: " + str(metrics.roc_auc_score(fullListOfTrueYOutcomeForAUCROCAndPR_list,
														fullListOfPredictedYProbsForAUCROC_list,
														average = 'weighted'))
	#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
	PRResults = metrics.precision_recall_fscore_support(fullListOfTrueYOutcomeForAUCROCAndPR_list,
														fullListOfPredictedYForPrecisionRecall_list,
														average = 'binary')
	print 'Precision: ' + str(PRResults[0])
	print 'Recall: ' + str(PRResults[1])
	print 'Binary F1 Score: ' + str(PRResults[2]) #FBeta score with beta = 1.0
	print 'Support: ' + str(PRResults[3])

	average_precision = metrics.average_precision_score(fullListOfTrueYOutcomeForAUCROCAndPR_list,
									   			       fullListOfPredictedYForPrecisionRecall_list)
	precision, recall, _ = metrics.precision_recall_curve(fullListOfTrueYOutcomeForAUCROCAndPR_list,
									   			       fullListOfPredictedYForPrecisionRecall_list)


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputFileNotes_test', type=str, default = '01.preprocessed_notes.npz-test')
	parser.add_argument('--inputFileDiagnoses_test', type=str, default='02.preprocessed_diagnoses_270.npz-test')
	parser.add_argument('--modelFile', type=str, default = '03.model_output.npz')
	parser.add_argument('--hiddenDimSize', type=str, default='[270]', help='Number of layers and their size - for example [100,200] refers to two layers with 100 and 200 nodes.')
	parser.add_argument('--batchSize', type=int, default=100, help='Batch size.')
	ARGStemp = parser.parse_args()
	hiddenDimSize = [int(strDim) for strDim in ARGStemp.hiddenDimSize[1:-1].split(',')]
	ARGStemp.hiddenDimSize = hiddenDimSize
	return ARGStemp

if __name__ == '__main__':
	global tPARAMS
	tPARAMS = OrderedDict()
	global ARGS
	ARGS = parse_arguments()

	testModel()