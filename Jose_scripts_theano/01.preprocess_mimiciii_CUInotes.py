#################################################################################################
# author: junio@usp.br - Jose F Rodrigues Jr
#SCRIPT preprocess_mimiciii_CUInotes.py
#INPUT: file "ADMISSIONS.csv" and file "CUINotes.csv"
#OUTPUT: file "preprocessed_notes.npz" => carries a serialized map in which each entry is a key value of
#the form (subject_id, [(hadm_id,admittime, [CUIsvector])], that is, given a patient id (subject_id),
#it returns the list of admissions of that patient, each one identified by its corresponding hadm_id,
#and time stamp; it also carries the vector containing all the CUIs for that admission - it is a map of list of lists.
#The admissions are temporally ordered.
#OUTPUT MEANING: in the jargon of machine learning it provides the X data for the training process
#COMMAND LINE: "python preprocess_mimiciii_notes.py"
#(provided all the files can be found and all the libraries are installed)
#################################################################################################

import pickle
from datetime import datetime
import argparse
import computeDistributions

global ARGS

def get_CUINotes_from_CSV_file(fileName):
	#by 'note', I mean note's vector
	#noteevents: https://mimic.physionet.org/mimictables/noteevents/
	mimicFile = open(fileName, 'r')
	mimicFile.readline()
	categories_map = {}
	descriptions_map = {}
	hadmToCUINotes_Map = {}
	inconsistent_hadm_ids = 0
	erroneous_records = 0
	invalid_CUINote = 0
	for line in mimicFile:			 #   rowid  , subject_id    ,    hadm_id   ,  .....
		tokens = line.strip().split(',')

		is_error_field = tokens[9]
		# A 1 in the ISERROR column indicates that a physician has identified this note as an error
		# check the mimic-iii website for details at https://mimic.physionet.org/mimictables/noteevents/
		if is_error_field == '1':
			#print('ERROR FIELD == 1')
			erroneous_records += 1
			continue

		# Collect some statistics
		#There are 15 different categories.
		#The most common are: nursing/other, radiology, nursing, ecg, physician, discharge summary, echo, respiratory,...
		category = tokens[6].lower().rstrip().strip('"')
		if category in categories_map:
			categories_map[category] += 1
		else:
			categories_map[category] = 1

		#There are 3.848 different descriptions.
		#The most common are: report, nursing progress note, chest, Physician Resident Progress, ...
		description = tokens[7].lower().rstrip().strip('"')
		if description in descriptions_map:
			descriptions_map[description] += 1
		else:
			descriptions_map[description] = 1

		#Observation: the category, description distribution is relevant
		#select category, description, count(*)
		#from noteevents
		#group by category, description
		#order by count(*) desc

		#--------------------------
		subject_id = tokens[1]
		hadm_id = tokens[2]
		if hadm_id == '':
			inconsistent_hadm_ids += 1
			continue
		else: hadm_id = int(hadm_id)
		#Read the CUIS, take off the quotes
		CUInote_vector = tokens[10].strip().strip('"')
		#Take off the 'C' before every code
		CUInote_vector = CUInote_vector.replace('C', '')

		# Here we can pick a category of interest
		#if category != 'physician':
		#	continue

		#collect the admission identification data, so that it is possible to match it with the diagnoses data
		if len(CUInote_vector) < 5:
			invalid_CUINote += 1
		else:
			if hadm_id in hadmToCUINotes_Map:
				hadmToCUINotes_Map[hadm_id].append(CUInote_vector)
			else:
				hadmToCUINotes_Map[hadm_id] = []
				hadmToCUINotes_Map[hadm_id].append(CUInote_vector)
	mimicFile.close()

	print('Number of note records with null hadm_ids: ' + str(inconsistent_hadm_ids))
	print('Number of erroneous records (ERROR = 1): ' + str(erroneous_records))
	print('Number of records with invalid CUI Notes: ' + str(invalid_CUINote))
	categories_map = sorted(categories_map.items(), key=lambda x: x[1], reverse=True)
	descriptions_map = sorted(descriptions_map.items(), key=lambda x: x[1], reverse=True)
	print('CATEGORIES: ' + str(categories_map))
	print('DESCRIPTIONS: ' + str(descriptions_map))
	return hadmToCUINotes_Map

def proccess_list_of_CUInotes(aListOfCUINotes):
	#split and convert to int
	#one admission has many CUINotes (each a string of ints), we "agregate" them by creating a set of CUIs with set union
	set_of_CUIcodes = set()
	for CUINotes in aListOfCUINotes:
		tokens = CUINotes.strip().split(' ')
		for value in tokens:
			if value == '"':
				print('no')
			set_of_CUIcodes.add(int(value))

	return list(set_of_CUIcodes)

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--admissions_file', type=str, default='/home/junio/Desktop/Ju/MIMIC-III/mimic-original/ADMISSIONS.csv')
	parser.add_argument('--notes_file', type=str, default='sample_2.csv')
	parser.add_argument('--output_file_name', type=str, default='01.preprocessed_notes')
	argsTemp = parser.parse_args()
	return argsTemp

if __name__ == '__main__':
	global ARGS
	ARGS = parse_arguments()
	mimic_ADMISSIONS_csv = open(ARGS.admissions_file, 'r')
	# row_id,subject_id,hadm_id,admittime,dischtime,deathtime,admission_type,admission_location,discharge_location,insurance,language,religion,marital_status,ethnicity,
	mimic_ADMISSIONS_csv.readline() # discard first line

	initial_number_of_admissions = 0
	subjectTOhadms_Map = {}
	hadmTOadmttime_Map = {}
	print('Build map subject_id -> list of hadm_ids')
	for line in mimic_ADMISSIONS_csv:
		initial_number_of_admissions += 1
		tokens = line.strip().split(',')
		subject_id = int(tokens[1])
		hadm_id = int(tokens[2])
		admittime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
		dischargetime = datetime.strptime(tokens[4], '%Y-%m-%d %H:%M:%S')

		# bypass temporally inconsistent admissions
		if admittime > dischargetime:
			continue

		#hadmTOadmttime_Map(hadm_id) -> time of admission hadm_id
		hadmTOadmttime_Map[hadm_id] = admittime

		#subjectTOhadms_Map(subject_id) -> set of hadms for subject_id
		if subject_id in subjectTOhadms_Map: subjectTOhadms_Map[subject_id].append(hadm_id)
		else: subjectTOhadms_Map[subject_id] = [hadm_id] #the brackets indicate that it will be a list
	mimic_ADMISSIONS_csv.close()
	print('-Initial number of admissions: ' + str(initial_number_of_admissions))
	print('-Initial number of subjects: ' + str(len(subjectTOhadms_Map)))

	#one line in the noteevents file contains only one note for one admission hadm_id
	#but one hadm_id has many noteevents; even multiple notevents of the same category
	print('Building Map: hadm_id to set of Notes from ' + ARGS.notes_file)
	hadmToCUINotes_Map = get_CUINotes_from_CSV_file(ARGS.notes_file)
	print('-Number of valid admissions (the ones with, at least, one note): ' + str(len(hadmToCUINotes_Map)))
	for hadm_id, notes_list in hadmToCUINotes_Map.items():
		hadmToCUINotes_Map[hadm_id] = proccess_list_of_CUInotes(hadmToCUINotes_Map[hadm_id])

	#Cleaning up inconsistencies
	#Some admissions do not have an associated text - we toss them off
	#this may cause the presence of patients (subject_ids) with 0 admissions hadm_id; we clear these guys too
	number_of_admissions_without_notes = 0
	print('Cleaning up admissions without notes')
	for subject_id, subjectHadmList in subjectTOhadms_Map.items():   #hadmTOadmttime_Map,subjectTOhadms_Map,hadm_cid9s_Map
		subjectHadmListCopy = list(subjectHadmList)    #copy the list, iterate over the copy, edit the original; otherwise, iteration problems
		for hadm_id in subjectHadmListCopy:
			if hadm_id not in hadmToCUINotes_Map.keys():  #map hadmToCUINotes_Map is already valid by creation
				number_of_admissions_without_notes += 1
				del hadmTOadmttime_Map[hadm_id]     #delete by key
				#since this hadm_id does not have a note, we remove it from the corresponding subject's list
				subjectHadmList.remove(hadm_id)
	print('-Number of admissions without notes: ' + str(number_of_admissions_without_notes))
	print('-Number of admissions after cleaning: ' + str(len(hadmToCUINotes_Map)))
	print('-Number of subjects after cleaning: ' + str(len(subjectTOhadms_Map)))

	#since the data in the database is not necessarily time-ordered
	#here we sort the admissions (hadm_id) according to the admission time (admittime)
	#after this, we have a list subjectTOorderedHADM_IDS_Map(subject_id) -> admission-time-ordered set of notes
	print('Building Map: subject_id to admission-ordered (admittime, Note) and cleaning one-admission-only patients')
	subjectTOorderedHADM_IDS_Map = {}
	#for each admission hadm_id of each patient subject_id
	number_of_subjects_with_less_than_two_admissions = 0
	subjects_in_map = list(subjectTOhadms_Map.keys())
	final_number_of_admissions = 0
	for subject_id in subjects_in_map:
		subjectHadmList = subjectTOhadms_Map[subject_id]
		if len(subjectHadmList) < 2:
			number_of_subjects_with_less_than_two_admissions += 1
			del subjectTOhadms_Map[subject_id]
			continue  #discard subjects with only one admission
		#sorts the hadm_ids according to date admttime
		#only for the hadm_id in the list hadmList
		sortedList = sorted([(hadm_id, hadmTOadmttime_Map[hadm_id], hadmToCUINotes_Map[hadm_id]) for hadm_id in subjectHadmList])
		# each element in subjectTOhadms_Map is a key-value (subject_id, [(hadm_id, admittime, floats_vector))]
		subjectTOhadms_Map[subject_id] = sortedList
		final_number_of_admissions += len(sortedList)
	print('-Number of discarded subjects with less than two admissions: ' + str(number_of_subjects_with_less_than_two_admissions))
	subjectTOorderedHADM_IDS_Map = subjectTOhadms_Map
	print('-Number of subjects after ordering: ' + str(len(subjectTOorderedHADM_IDS_Map)))

	CUI_ordered_internalCodesMap = {}
	CODES_distributionMAP = computeDistributions.writeDistributions(all_subjects_map_of_admissionData = subjectTOorderedHADM_IDS_Map)
	for i, key in enumerate(CODES_distributionMAP):
		CUI_ordered_internalCodesMap[key[0]] = i

	print('Converting database CUI ids to sequential integer ids')
	# each element in subjectTOorderedHADM_IDS_Map is a key-value (subject_id, [(hadm_id, admittime, ICD9_List)])
	final_number_of_admissions = 0
	for subject, admissions in subjectTOorderedHADM_IDS_Map.items():
		for admission in admissions:
			final_number_of_admissions += 1
			codes_list = admission[2]
			for i in range(len(codes_list)):
				codes_list[i] = CUI_ordered_internalCodesMap[codes_list[i]]  # alter the code number to an internal sequential list

	#writing all the data
	print('Writing patients'' notes read from files ' + ARGS.notes_file)
	pickle.dump(subjectTOorderedHADM_IDS_Map, open(ARGS.output_file_name + '.npz', 'wb'), protocol = 2)
	print('-Final number of subjects'' notes for training: ' + str(len(subjectTOorderedHADM_IDS_Map)))
	print('-Final number of admissions for training: ' + str(final_number_of_admissions))