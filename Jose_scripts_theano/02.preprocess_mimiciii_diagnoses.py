#################################################################################################
# author: junio@usp.br - Jose F Rodrigues Jr
#SCRIPT preprocess_mimiciii_diagnoses.py
#INPUT: file "ADMISSIONS.csv" and file "DIAGNOSES_ICD.csv"
#OUTPUT: file "preprocessed_diagnoses_270.npz" => carries a serialized map in which each entry is a key
#value of the form (subject_id, [(hadm_id, admittime, diagnoses_list))], that is, given a patient id
#(subject_id), it returns the list of admissions of that patient, each one identified by its corresponding
#hadm_id, and time stamp; it also carries the list of diagnoses corresponding to that admission.
#The admissions are temporally ordered.
#OUTPUT MEANING: in the jargon of machine learning it provides the Y data for the training process,
#the "label" of the process
#COMMAND LINE: "python preprocess_mimiciii_diagnoses.py" (provided all the files can be found and all
#the libraries are installed)
#################################################################################################

import sys
import pickle
from datetime import datetime
import argparse
import computeDistributions

global ARGS


# given a map of (hadm_id, set of diagnoses icd9 codes), convert the map to (hadm_id, CCS codes)
def map_ICD9_to_CCS(map):
    icd9TOCCS_Map = pickle.load(open(sys.path[0] + '/icd9_to_css_dictionary', 'rb'))
    procCODEstoInternalID_map = {}
    set_of_used_codes = set()
    for (hadm_id, ICD9s_List) in map.items():
        for ICD9 in ICD9s_List:
            while (len(ICD9) < 6): ICD9 += ' '  # pad right white spaces because the CCS mapping uses this pattern
            try:
                CCS_code = icd9TOCCS_Map[ICD9]
                if hadm_id in procCODEstoInternalID_map:
                    procCODEstoInternalID_map[hadm_id].append(CCS_code)
                else:
                    procCODEstoInternalID_map[hadm_id] = [CCS_code]
                set_of_used_codes.add(ICD9)
            except KeyError:
                print(str(sys.exc_info()[0]) + '  ' + str(ICD9) + ". ICD9 code not found, please verify your ICD9 to CCS mapping before proceeding.")
    print('-Total number (complete set) of ICD9 codes (diag + proc): ' + str(len(set(icd9TOCCS_Map.keys()))))
    print('-Total number (complete set) of CCS codes (diag + proc): ' + str(len(set(icd9TOCCS_Map.values()))))
    print('-Total number of ICD9 codes actually used: ' + str(len(set_of_used_codes)))

    return procCODEstoInternalID_map


def get_ICD9s_from_mimic_file(fileName, hadmToMap):
    mimicFile = open(fileName, 'r')  # row_id,subject_id,hadm_id,seq_num,ICD9_code
    mimicFile.readline()
    number_of_null_ICD9_codes = 0
    for line in mimicFile:  # 0  ,     1    ,    2   ,   3  ,    4
        tokens = line.strip().split(',')
        hadm_id = int(tokens[2])
        if (len(tokens[4]) == 0):  # ignore diagnoses where ICD9_code is null
            number_of_null_ICD9_codes += 1
            continue;

        ICD9_code = tokens[4]
        if ICD9_code.find("\"") != -1:
            ICD9_code = ICD9_code[1:-1]  # toss off quotes and proceed
        # since diagnosis and procedure ICD9 codes have intersections, a prefix is necessary for disambiguation
        if fileName == ARGS.diagnoses_file:
            ICD9_code = 'D' + ICD9_code
        else:
            ICD9_code = 'P' + ICD9_code
        # To understand the line below, check https://mimic.physionet.org/mimictables/diagnoses_icd/
        # "The code field for the ICD-9-CM Principal and Other Diagnosis Codes is six characters in length (not really!),
        # with the decimal point implied between the third and fourth digit for all diagnosis codes other than the V codes.
        # The decimal is implied for V codes between the second and third digit."
        # Actually, if you look at the codes (https://raw.githubusercontent.com/drobbins/ICD9/master/ICD9.txt), simply take the three first characters
        if not ARGS.map_ICD9_to_CCS:
            ICD9_code = ICD9_code[:4]  # No CCS mapping, get the first alphanumeric four letters only

        if hadm_id in hadmToMap:
            hadmToMap[hadm_id].add(ICD9_code)
        else:
            hadmToMap[hadm_id] = set()  # use set to avoid repetitions
            hadmToMap[hadm_id].add(ICD9_code)
    for hadm_id in hadmToMap.keys():
        hadmToMap[hadm_id] = list(hadmToMap[hadm_id])  # convert to list, as the rest of the codes expects
    mimicFile.close()
    print('-Number of null ICD9 codes in file ' + fileName + ': ' + str(number_of_null_ICD9_codes))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--admissions_file', type=str, default='/home/junio/Desktop/Ju/MIMIC-III/mimic-original/ADMISSIONS.csv')
    parser.add_argument('--diagnoses_file', type=str, default='/home/junio/Desktop/Ju/MIMIC-III/mimic-original/DIAGNOSES_ICD.csv', help='The DIAGNOSES_ICD.csv file of mimic-iii distribution.')
    parser.add_argument('--procedures_file', type=str, default='', help='The optional PROCEDURES_ICD.csv file of mimic-iii distribution - for processing using procedures codes.')
    parser.add_argument('--output_file_name', type=str, default='02.preprocessed_diagnoses', help='The output file name.')
    parser.add_argument('--map_ICD9_to_CCS', type=int, default=1, choices=[0, 1], help='False/True [0/1] to convert ICD9 codes to CCS codes (better accuracy, less granularity); refer to https://www.hcup-us.ahrq.gov/toolssoftware/CCS/CCS.jsp.')
    argsTemp = parser.parse_args()
    return argsTemp


if __name__ == '__main__':
    global ARGS
    ARGS = parse_arguments()

    # one line of the admissions file contains one admission hadm_id of one subject_id at a given time admittime
    print('Building Maps: hadm_id to admtttime; and Map: subject_id to set of all its hadm_ids')
    subjectTOhadms_Map = {}
    hadmTOadmttime_Map = {}  # 0  ,     1    ,    2  ,     3   ,    4
    mimic_ADMISSIONS_csv = open(ARGS.admissions_file, 'r')
    # row_id,subject_id,hadm_id,admittime,dischtime,deathtime,admission_type,admission_location,discharge_location,insurance,language,religion,marital_status,ethnicity,
    mimic_ADMISSIONS_csv.readline()

    initial_number_of_admissions = 0
    for line in mimic_ADMISSIONS_csv:
        initial_number_of_admissions += 1
        tokens = line.strip().split(',')
        subject_id = int(tokens[1])
        hadm_id = int(tokens[2])
        admittime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
        dischargetime = datetime.strptime(tokens[4], '%Y-%m-%d %H:%M:%S')

        temp = dischargetime - admittime
        if admittime > dischargetime: continue

        # the time-related stuff that follows, assumes that the ADMISSIONS.csv file
        # was exported from postgres with command
        # \copy (SELECT * FROM ADMISSIONS ORDER BY SUBJECT_ID, ADMITTIME ASC) TO '/j/dat/mimic-original/ADMISSIONS.csv' WITH (FORMAT CSV,HEADER TRUE)

        # keep track of the admission amount of time
        # hadmTOadmttime_Map(hadm_id) -> time of admission hadm_id
        hadmTOadmttime_Map[hadm_id] = admittime

        previous_admission = hadm_id
        previous_subject = subject_id

        # subjectTOhadms_Map(subject_id) -> set of hadms for subject_id
        if subject_id in subjectTOhadms_Map:
            subjectTOhadms_Map[subject_id].append(hadm_id)
        else:
            subjectTOhadms_Map[subject_id] = [hadm_id]  # the brackets indicate that it will be a list

    mimic_ADMISSIONS_csv.close()
    print('-Initial number of admissions: ' + str(initial_number_of_admissions))
    print('-Initial number of subjects: ' + str(len(subjectTOhadms_Map)))
    hadmToICD9CODEs_Map = {}
    hadmToICD9ProcCODEs_Map = {}

    if len(ARGS.diagnoses_file) > 0:
        # one line in the diagnoses file contains only one diagnose code (ICD9) for one admission hadm_id
        print('Building Map: hadm_id to set of ICD9 codes from DIAGNOSES_ICD')
        get_ICD9s_from_mimic_file(ARGS.diagnoses_file, hadmToICD9CODEs_Map)
    if len(ARGS.procedures_file) > 0:
        print('Building Map: hadm_id to set of ICD9 codes from PROCEDURES_ICD')
        get_ICD9s_from_mimic_file(ARGS.procedures_file, hadmToICD9ProcCODEs_Map)

    print('-Number of valid admissions (at least one diagnosis): ' + str(len(hadmToICD9CODEs_Map)))

    # Here we make sure we have only admissions with procedures AND diagnosis
    if len(ARGS.procedures_file) > 0:
        print('-Number of procedures codes before cleaning: ' + str(len(hadmToICD9ProcCODEs_Map)))
        for hadm_id in hadmToICD9CODEs_Map.keys():
            if hadm_id not in hadmToICD9ProcCODEs_Map.keys():
                del hadmToICD9CODEs_Map[hadm_id]
        for hadm_id in hadmToICD9ProcCODEs_Map.keys():
            if hadm_id not in hadmToICD9CODEs_Map.keys():
                del hadmToICD9ProcCODEs_Map[hadm_id]
        print('-Number of procedures after cleaning: ' + str(len(hadmToICD9ProcCODEs_Map)))

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
        for hadm_id in hadmListCopy:
            if hadm_id not in hadmToICD9CODEs_Map.keys():  # map hadmToICD9CODEs_Map is already valid by creation
                number_of_admissions_without_diagnosis += 1
                del hadmTOadmttime_Map[hadm_id]  # delete by key
                hadmList.remove(hadm_id)
        if len(hadmList) == 0:  # toss off subject_id without admissions
            number_of_subjects_without_valid_admissions += 1
            subjects_without_admission.append(subject_id)
    for subject in subjects_without_admission:
        del subjectTOhadms_Map[subject] # delete by value

    print('-Number of admissions without diagnosis: ' + str(number_of_admissions_without_diagnosis))
    print('-Number of admissions after cleaning: ' + str(len(hadmToICD9CODEs_Map)))
    print('-Number of subjects without admissions: ' + str(number_of_subjects_without_valid_admissions))
    print('-Number of subjects after cleaning: ' + str(len(subjectTOhadms_Map)))

    if ARGS.map_ICD9_to_CCS:
        print('Mapping ICD9 codes to CCS codes')
        hadmToICD9CODEs_Map = map_ICD9_to_CCS(hadmToICD9CODEs_Map)
        if len(ARGS.procedures_file) > 0:
            hadmToICD9ProcCODEs_Map = map_ICD9_to_CCS(hadmToICD9ProcCODEs_Map)

    # since the data in the database is not necessarily time-ordered
    # here we sort the admissions (hadm_id) according to the admission time (admittime)
    # after this, we have a list subjectTOorderedHADM_IDS_Map(subject_id) -> admission-time-ordered set of ICD9 codes
    print('Building Map: subject_id to admission-ordered (admittime, ICD9s set) and cleaning one-admission-only patients')
    subjectTOorderedHADM_IDS_Map = {}
    subjectTOProcHADM_IDs_Map = {}
    # for each admission hadm_id of each patient subject_id
    number_of_subjects_with_only_one_visit = 0
    for subject_id, hadmList in subjectTOhadms_Map.items():
        if len(hadmList) < 2:
            number_of_subjects_with_only_one_visit += 1
            continue  # discard subjects with only 2 admissions
        # sorts the hadm_ids according to date admttime
        # only for the hadm_id in the list hadmList
        sortedList = sorted([(hadm_id, hadmTOadmttime_Map[hadm_id], hadmToICD9CODEs_Map[hadm_id]) for hadm_id in hadmList])
        # each element in subjectTOorderedHADM_IDS_Map is a key-value (subject_id, [(hadm_id, admittime, ICD9_List)])
        subjectTOorderedHADM_IDS_Map[subject_id] = sortedList
    print('-Number of discarded subjects with only one admission: ' + str(number_of_subjects_with_only_one_visit))
    print('-Number of subjects after ordering: ' + str(len(subjectTOorderedHADM_IDS_Map)))

    CCS_ordered_internalCodesMap = {}
    CODES_distributionMAP = computeDistributions.writeDistributions(ARGS.admissions_file, hadmToICD9CODEs_Map, subjectTOhadms_Map, subjectTOorderedHADM_IDS_Map)
    for i, key in enumerate(CODES_distributionMAP):
        CCS_ordered_internalCodesMap[key[0]] = i

    # print(distribution of CCS codes
    if ARGS.map_ICD9_to_CCS:
        CCS_to_descriptionMap = pickle.load(open(sys.path[0] + '/ccs_to_description_dictionary', 'rb'))
        i = 0
        internalCodeToTextualDescriptionMAP = {}
        for CODE, value in CODES_distributionMAP:
            print('Internal code: ' + str(i) + '; CCS code: ' + str(CODE) + '; Frequency: ' + str(value) + '; Textual: ' + CCS_to_descriptionMap[CODE])
            internalCodeToTextualDescriptionMAP[i] = CCS_to_descriptionMap[CODE]
            i += 1
        pickle.dump(internalCodeToTextualDescriptionMAP, open('internalCodeToTextualDescriptionMAP.pickle', 'wb'), protocol = 2)

    print('Converting database ids to sequential integer ids')
    # each element in subjectTOorderedHADM_IDS_Map is a key-value (subject_id, [(hadm_id, admittime, ICD9_List)])
    final_number_of_admissions = 0
    for subject, admissions in subjectTOorderedHADM_IDS_Map.items():
        for admission in admissions:
            final_number_of_admissions += 1
            codes_list = admission[2]
            for i in range(len(codes_list)):
                codes_list[i] = CCS_ordered_internalCodesMap[codes_list[i]] #alter the code number to an internal sequential list

    print('')
    nCodes = len(CCS_ordered_internalCodesMap)
    print('-Number of actually used DIAGNOSES codes: ' + str(nCodes))

    print('Writing patients'' diagnoses read from files ' + ARGS.diagnoses_file)
    pickle.dump(subjectTOorderedHADM_IDS_Map, open(ARGS.output_file_name + '_' + str(nCodes) + '.npz', 'wb'), protocol = 2)
    print('-Final number of subjects'' diagnoses for training: ' + str(len(subjectTOorderedHADM_IDS_Map)))
    print('-Final number of admissions for training: ' + str(final_number_of_admissions))