#!/usr/bin/python
# Code previously taken from junio@usp.br - Jose F Rodrigues Jr (special thanks)
# Code modified by jamil.zaghir@grenoble-inp.org - Jamil Zaghir

import pickle
from datetime import datetime
import argparse
import math
import icd9_to_ccs as ccsMapper
import copy

global ARGS


CUI_set = set()
CCS_set = set()

def get_ICD9s_from_mimic_file(fileName, hadmToMap):
        mimicFile = open(fileName, 'r')  # row_id,subject_id,hadm_id,seq_num,ICD9_code
        mimicFile.readline()
        number_of_null_ICD9_codes = 0
        for line in mimicFile:  # 0  ,     1    ,    2   ,   3  ,    4
                tokens = line.strip().split(',')
                hadm_id = int(tokens[2])
                if (len(tokens[4]) == 0):  # ignore diagnoses where ICD9_code is null
                        number_of_null_ICD9_codes += 1
                        continue

                ICD9_code = tokens[4]
                if ICD9_code.find("\"") != -1:
                        ICD9_code = ICD9_code[1:-1]  # toss off quotes and proceed
                ICD9_code = ICD9_code
        # To understand the line below, check https://mimic.physionet.org/mimictables/diagnoses_icd/
        # "The code field for the ICD-9-CM Principal and Other Diagnosis Codes is six characters in length (not really!),
        # with the decimal point implied between the third and fourth digit for all diagnosis codes other than the V codes.
        # The decimal is implied for V codes between the second and third digit."
        # Actually, if you look at the codes (https://raw.githubusercontent.com/drobbins/ICD9/master/ICD9.txt), simply take the three first characters
        # if not ARGS.map_ICD9_to_CCS:
        #     ICD9_code = ICD9_code[:4]  # No CCS mapping, get the first alphanumeric four letters only
                if hadm_id in hadmToMap:
                        hadmToMap[hadm_id].add(ICD9_code)
                else:
                        hadmToMap[hadm_id] = set()  # use set to avoid repetitions
                        hadmToMap[hadm_id].add(ICD9_code)
        for hadm_id in hadmToMap.keys():
                hadmToMap[hadm_id] = list(hadmToMap[hadm_id])  # convert to list, as the rest of the codes expects
        mimicFile.close()
        print('-Number of null ICD9 codes in file ' + fileName + ': ' + str(number_of_null_ICD9_codes))
        return hadmToMap


def get_CUINotes_from_CSV_file(fileName):
        """ Get the mapping ADM_ID -> vector of CUI codes (vector of string) with 'C' tossed off
        Get others statistics such as number of occurrence of each category and
        description, number of erroneous records (error = 1), and invalid ones."""
        # one line in the noteevents file contains one note for one admission only
        # however one admission has many notes; even multiple notes of the same category
        mimicFile = open(fileName, 'r')
        mimicFile.readline()
        categories_map = {}
        descriptions_map = {}
        hadmToCUINotes_Map = {}
        inconsistent_hadm_ids = 0
        erroneous_records = 0
        invalid_CUINote = 0
        for line in mimicFile:                   #   rowid  , subject_id    ,    hadm_id   ,  .....
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
                # if category != 'radiology':
                #       continue

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


def split_and_convertToInt(aListOfCUINotes):
        """ Split vector of CUIs and convert them to int (in a set to avoid duplicate) """
        # One admission has many CUINotes (each a string of ints), we "agregate" them by creating a set of CUIs with set union
        set_of_CUIcodes = set()
        for CUINotes in aListOfCUINotes:
                tokens = CUINotes.strip().split(' ')
                for value in tokens:
                        if value == '"':
                                print('no')
                        set_of_CUIcodes.add(int(value))
                        CUI_set.add(int(value)) # Add the CUI in the set of ALL possible CUIs in the file
        return list(set_of_CUIcodes)


def getCUICodes_givenAdmID():
        print('Building Map: hadm_id to set of Notes from ' + ARGS.notes_file)
        hadmToCUINotes_Map = get_CUINotes_from_CSV_file(ARGS.notes_file)
        print('-Number of valid admissions (the ones with, at least, one note): ' + str(len(hadmToCUINotes_Map)))
        for hadm_id, notes_list in hadmToCUINotes_Map.items():
                hadmToCUINotes_Map[hadm_id] = split_and_convertToInt(hadmToCUINotes_Map[hadm_id])
        return hadmToCUINotes_Map


def admissionsParser():
        """ Get three maps:
        1- Patient_ID -> Set of ADM_ID
        2- ADM_ID -> ADM_TIME
        3- ADM_ID -> DISCH_TIME
        4- Patient_ID -> DEATH_TIME (null if not passed away)
        """
        mimic_ADMISSIONS_csv = open(ARGS.admissions_file, 'r')
        mimic_ADMISSIONS_csv.readline() # discard the header of the csv file
        initial_number_of_admissions = 0
        subjectTOhadms_Map = {}
        hadmTOadmttime_Map = {}
        hadmTOdischtime_Map = {}
        print('Build map subject_id -> list of hadm_ids')
        for line in mimic_ADMISSIONS_csv:
            initial_number_of_admissions += 1
            tokens = line.strip().split(',')
            subject_id = int(tokens[1])
            hadm_id = int(tokens[2])
            admittime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
            dischargetime = datetime.strptime(tokens[4], '%Y-%m-%d %H:%M:%S')
            # inconsistent admissions to avoid
            if admittime > dischargetime:
                continue
            # update the maps
            hadmTOadmttime_Map[hadm_id] = admittime
            hadmTOdischtime_Map[hadm_id] = dischargetime
            if subject_id in subjectTOhadms_Map:
                subjectTOhadms_Map[subject_id].append(hadm_id)
            else:
                subjectTOhadms_Map[subject_id] = [hadm_id] #the brackets indicate that it will be a list
        mimic_ADMISSIONS_csv.close()
        print('-Initial number of admissions: ' + str(initial_number_of_admissions))
        print('-Initial number of subjects: ' + str(len(subjectTOhadms_Map)))
        return subjectTOhadms_Map, hadmTOadmttime_Map, hadmTOdischtime_Map


def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument('--admissions_file', type=str, default='/home/jamilz/LIG/MIMIC III/mimic-iii-clinical-database-1.4/ADMISSIONS.csv')
        parser.add_argument('--diagnoses_file', type=str, default='/home/jamilz/LIG/MIMIC III/mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv')
        parser.add_argument('--notes_file', type=str, default='output.csv')
        parser.add_argument('--output_file_name', type=str, default='prepared_data')
        argsTemp = parser.parse_args()
        return argsTemp


if __name__ == '__main__':
        # Step 0 : get arguments (admissions file, diagnoses file, CUIs file, and output filename)
        global ARGS
        ARGS = parse_arguments()

        # Step 1 : get admissions ID its admittime given the patient ID
        subjectTOhadms_Map, hadmTOadmttime_Map, hadmTOdischtime_Map = admissionsParser()

        # Step 2 : get CUIs from the CUIs file given admission ID
        hadmToCUINotes_Map = getCUICodes_givenAdmID()

        # Step 3 : Remove admissions ID whose notes is empty, remove patients ID if there isn't any note anymore.
        # Some admissions do not have an associated text - we toss them off
        # May cause the presence of patients with 0 admissions hadm_id; we clear these patients too
        number_of_admissions_without_notes = 0
        print('Cleaning up admissions without notes')
        for subject_id, subjectHadmList in subjectTOhadms_Map.items():   # hadmTOadmttime_Map,subjectTOhadms_Map,hadm_cid9s_Map
                subjectHadmListCopy = list(subjectHadmList)    # Copy the list, iterate over the copy, edit the original; otherwise, iteration problems
                for hadm_id in subjectHadmListCopy:
                        if hadm_id not in hadmToCUINotes_Map.keys():  # Map hadmToCUINotes_Map is already valid by creation
                                number_of_admissions_without_notes += 1
                                del hadmTOadmttime_Map[hadm_id]  # Delete by key
                                # Since this hadm_id does not have a note, we remove it from the corresponding subject's list
                                subjectHadmList.remove(hadm_id)
        print('-Number of admissions without notes: ' + str(number_of_admissions_without_notes))
        print('-Number of admissions after cleaning: ' + str(len(hadmToCUINotes_Map)))
        print('-Number of subjects after cleaning: ' + str(len(subjectTOhadms_Map)))

        # Step 4 : get ICDs from the Diagnoses file given admission ID
        hadmToICD9CODEs_Map = {}

        if len(ARGS.diagnoses_file) > 0:
                # one line in the diagnoses file contains only one diagnose code (ICD9) for one admission hadm_id
                print('Building Map: hadm_id to set of ICD9 codes from DIAGNOSES_ICD')
                # get_ICD9s_from_mimic_file(ARGS.diagnoses_file, hadmToICD9CODEs_Map)
                hadmToICD9CODEs_Map = get_ICD9s_from_mimic_file(ARGS.diagnoses_file, hadmToICD9CODEs_Map)

        print('-Number of valid admissions (at least one diagnosis): ' + str(len(hadmToICD9CODEs_Map)))

    # Cleaning up inconsistencies
    # some tuples in the diagnoses table have ICD9 empty; we clear the admissions without diagnoses from all the maps
    # this may cause the presence of patients (subject_ids) with 0 admissions hadm_id; we clear these guys too
    # We also clean admissions in which admission time < discharge time - there are 89 records like that in the original dataset
        number_of_admissions_without_diagnosis = 0
        number_of_subjects_without_valid_admissions = 0
        print('Cleaning up admissions without diagnoses')
        subjects_without_admission = []
        for subject_id, hadmList in subjectTOhadms_Map.items():  # hadmTOadmttime_Map,subjectTOhadms_Map,hadm_cid9s_Map
                hadmListCopy = list(hadmList)  # copy the list, iterate over the copy, edit the original; otherwise, iteration problems
                for admission in hadmListCopy:
                        if admission not in hadmToICD9CODEs_Map.keys():  # map hadmToICD9CODEs_Map is already valid by creation
                                number_of_admissions_without_diagnosis += 1
                                del hadmTOadmttime_Map[admission]  # delete by key
                                hadmList.remove(admission)
                        if len(hadmList) == 0:  # toss off subject_id without admissions
                                number_of_subjects_without_valid_admissions += 1
                                subjects_without_admission.append(subject_id)
        for subject in subjects_without_admission:
                del subjectTOhadms_Map[subject] # delete by value

        print('-Number of admissions without diagnosis: ' + str(number_of_admissions_without_diagnosis))
        print('-Number of admissions after cleaning: ' + str(len(hadmToICD9CODEs_Map)))
        print('-Number of subjects without admissions: ' + str(number_of_subjects_without_valid_admissions))
        print('-Number of subjects after cleaning: ' + str(len(subjectTOhadms_Map)))

        # Step 5 : ordering admissions by admittime for each patient
        # We sort the admissions (hadm_id) according to the admission time (admittime)
        # After this, we have a list subjectTOorderedHADM_IDS_Map(subject_id) -> admission-time-ordered set of notes
        print('Building Map: subject_id to admission-ordered (admittime, Note) and cleaning one-admission-only patients')
        subjectTOorderedHADM_IDS_Map = {}
        # For each admission hadm_id of each patient subject_id
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

        # Step 6 : CUIs to integer IDs
        CUI_ordered_internalCodesMap = {}
        for i, key in enumerate(CUI_set):
                CUI_ordered_internalCodesMap[key] = i
        # print("Example : C0015726 -", CUI_ordered_internalCodesMap[15726])
        print('Converting database CUI ids to sequential integer ids')
        # each element in subjectTOorderedHADM_IDS_Map is a key-value (subject_id, [(hadm_id, admittime, CUI_List)])
        # final_number_of_admissions = 0
        for subject, admissions in subjectTOorderedHADM_IDS_Map.items():
                for admission in admissions:
                        # final_number_of_admissions += 1
                        codes_list = admission[2]
                        for i in range(len(codes_list)):
                                codes_list[i] = CUI_ordered_internalCodesMap[codes_list[i]]  # alter the code number to an internal sequential list

        # Step 7 : convert them to CCS code to reduce granularities and add them to subjectTOorderedHADM_IDS_Map
        # Get CCS codes
        icdtoccs_dico = dict()
        for k, icd_values in hadmToICD9CODEs_Map.items():
                for icd9code in icd_values:
                        if icd9code not in icdtoccs_dico.keys():
                                ccs_code = ccsMapper.getCCS(icd9code)
                                icdtoccs_dico[icd9code] = ccs_code
                                CCS_set.add(ccs_code)
        # Add CCS vectors, we now have : [subjID, [AdmID, AdmTime, [CUI], [CCS]]]
        subjectTOorderedHADM_IDS_COPY = copy.deepcopy(subjectTOorderedHADM_IDS_Map) # We need to copy, otherwise, iterations problems
        for subj, admlist in subjectTOorderedHADM_IDS_COPY.items():
                for adm_tuple in admlist:
                        adm_list = list(adm_tuple)
                        subjectTOorderedHADM_IDS_Map[subj].remove(adm_tuple)
                        ccs_list = list()
                        for icdcode in hadmToICD9CODEs_Map[adm_tuple[0]]:
                                ccs_list.append(icdtoccs_dico[icdcode])
                        adm_list.append(ccs_list)
                        adm_tuple = tuple(adm_list)
                        subjectTOorderedHADM_IDS_Map[subj].append(adm_tuple)
                        
        # Step 8 : CCS to integer IDs
        CCS_ordered_internalCodesMap = {}
        for i, key in enumerate(CCS_set):
            CCS_ordered_internalCodesMap[key] = i
        print('Converting database CCS ids to sequential integer ids')
        final_number_of_admissions = 0
        for subject, admissions in subjectTOorderedHADM_IDS_Map.items():
            for admission in admissions:
                final_number_of_admissions += 1
                codes_list = admission[3]
                for i in range(len(codes_list)):
                    codes_list[i] = CCS_ordered_internalCodesMap[codes_list[i]] # alter the code number to an internal sequential list
        
        # Step 9 : Add dischtime so it becomes: SubjectID -> [admID, admittime, [CUI], [CCS], dischtime]
        # Deep copy to avoid iter pb
        subjectTOorderedHADM_IDS_COPY = copy.deepcopy(subjectTOorderedHADM_IDS_Map)
        for subj, admlist in subjectTOorderedHADM_IDS_COPY.items():
            for adm_tuple in admlist:
                adm_list = list(adm_tuple)
                subjectTOorderedHADM_IDS_Map[subj].remove(adm_tuple)
                adm_list.append(hadmTOdischtime_Map[adm_tuple[0]])
                adm_tuple= tuple(adm_list)
                subjectTOorderedHADM_IDS_Map[subj].append(adm_tuple)

        # Step 10 : writing all the data
        print('Writing patients'' notes read from files ' + ARGS.notes_file)
        pickle.dump(subjectTOorderedHADM_IDS_Map, open(ARGS.output_file_name + '.npz', 'wb'), protocol = 2)
        print('-Final number of subjects'' notes for training: ' + str(len(subjectTOorderedHADM_IDS_Map)))
        print('-Final number of admissions for training: ' + str(final_number_of_admissions))
