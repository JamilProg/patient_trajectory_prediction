from quickumls import QuickUMLS
import csv, os, sys, time, re
from multiprocessing import Pool
import dill
import argparse

# Global variables
# QuickUMLS object
# matcher = QuickUMLS(quickumls_fp='./QuickUMLS', overlapping_criteria='score', threshold=0.7, similarity_name='cosine', window=5)
# TUI list
TUI_beta = ["T195","T123","T122","T103","T120","T200","T126","T116","T196","T131","T125","T129","T130","T121","T192","T127","T104","T114","T197","T109","T038","T034","T070","T067","T068","T069","T043","T201","T045","T041","T032","T040","T042","T039","T044","T020","T190","T049","T019","T047","T050","T033","T037","T048","T191","T046","T184","T091","T090","T017","T029","T023","T030","T031","T022","T025","T026","T018","T021","T024","T079","T203","T074","T075","T100","T011","T008","T194","T007","T012","T204","T099","T013","T004","T096","T016","T015","T001","T101","T098","T097","T014","T010","T005","T058","T060","T061"]
TUI_alpha = ["T020","T190","T049","T019","T047","T050","T033","T037","T048","T191","T046","T184","T038","T069","T068","T034","T070","T067","T043","T201","T045","T041","T044","T032","T040","T042","T039","T116","T195","T123","T122","T103","T120","T104","T200","T196","T126","T131","T125","T129","T130","T197","T114","T109","T121","T192","T127"]
# Input and output path
dirchunks = "./data/chunkssmall/"
diroutputchunks = "./data/outputchunkssmall/"


def main_funct(file):
    # Start the program
    list_cui = []
    list_terms = []
    lineNb = 0
    filename = dirchunks+file
    with open(filename, 'r') as fd:
        print("File", filename, "opened!")
        # Preparing outputfile
        outputFile = diroutputchunks+file+".output"
        fw = open(outputFile, 'w')
        for line in fd.readlines():
            # Keep IDs and non-text information
            count_comma = line.count(',')
            count_quote = line.count('"')
            if count_comma >= 1 :
                # New clinical note
                list_cui = []
                list_terms = []
                fw.write(line)
                continue
            # print(lineNb, flush=True)
            matches = matcher.match(line, best_match=True, ignore_syntax=False)
            concepts_output = []
            for phrase_candidate in matches:
                # Find max
                max=0
                for candidate in phrase_candidate:
                    if candidate['similarity']>max and set(candidate['semtypes']).intersection(TUIs):
                        max=candidate['similarity']
                # Get preferred terms for that max
                list_to_write = []
                if max >= 0:
                    for candidate in phrase_candidate:
                        if candidate['similarity']==max:
                            if candidate['term'] not in list_terms:
                                if candidate['cui'] not in list_cui :
                                    list_cui.append(candidate['cui'])
                                    list_terms.append(candidate['term'])
                                    list_to_write.append(candidate['cui'])
                concepts_output.append(list_to_write)
                resultline = ""
                for concepts in concepts_output:
                    for terms in concepts:
                        terms = re.sub(r' ', '', terms)
                        resultline += terms + " "
                if resultline is not "" :
                    fw.write(resultline+'\n')
                concepts_output = []
            lineNb+=1
            if count_quote >= 1 :
                # End of clinical note
                fw.write('"\n')
        fw.close()
        print("File", filename, "closed!")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--TUI', type=str, default='Beta', help='Choose Alpha or Beta as TUI filter - default : Beta')
    parser.add_argument('--t', type=float, default=0.9, help='Select a threshold (between 0 and 1) - default: 0.9')

    ARGStemp = parser.parse_args()
    return ARGStemp


if __name__ == "__main__":
    # Parse arguments
    global ARGS
    ARGS = parse_arguments()

    # Start the process
    global matcher
    matcher = QuickUMLS(quickumls_fp='./QuickUMLS', overlapping_criteria='score', threshold=ARGS.t, similarity_name='cosine', window=5)
    print("QuickUMLS Threshold: ", ARGS.t)
    global TUIs
    if ARGS.TUI == "Alpha" or ARGS.TUI == "alpha":
        TUIs = TUI_alpha
        print("TUI list Alpha selected")
    else:
        TUIs= TUI_beta
        print("TUI list Beta selected")
    pool = Pool(os.cpu_count()-4)
    pool.map(main_funct, os.listdir(dirchunks))
