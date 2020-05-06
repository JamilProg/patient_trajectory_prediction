#!/usr/bin/python
import re
import nltk
nltk.download('punkt')

def get_F1_score(P, R):
    """ compute F1-Score """
    if P+R == 0:
        return 0
    return ((2*P*R)/(P+R))


def compute_precision(outputfile, CUIs_to_find):
    """ compute Precision for the given file, given the CUI list to be found """
    total = 0
    found = 0
    with open(outputfile) as fp:
        while True:
            line = fp.readline()
            if not line :
                break
            cui_list = nltk.word_tokenize(line)
            for token in cui_list:
                total += 1
                if token in CUIs_to_find:
                    found += 1
    if total == 0:
        return 1
    precision = found / total
    # print ("P = " + str(found) + " / " + str(total))
    return precision


def compute_recall(outputfile, CUIs_to_find):
    """ compute recall for the given file, given the CUI list to be found """
    total = len(CUIs_to_find)
    if total == 0:
        return 1
    found = 0
    with open(outputfile) as fp:
        while True:
            line = fp.readline()
            if not line :
                break
            cui_list = nltk.word_tokenize(line)
            for token in cui_list:
                if token in CUIs_to_find:
                    found += 1
                    CUIs_to_find.remove(token)
    recall = found / total
    # print ("R = " + str(found) + " / " + str(total))
    return recall


def get_CUI_to_find(filename):
    """ get CUI list from labeled data """
    cuilist = []
    with open(filename) as fp:
        while True:
            line = fp.readline()
            if not line :
                break
            # m = re.search(r'(\|\|C\d+\|\|)', line)
            m = re.search(r'(\|\|[^0-9].+?\|\|)', line)
            if m:
                result = m.group()
                result = re.sub(r'\|\|(.+)\|\|', r'\1', result)
                # print(result)
                cuilist.append(result)
            # else:
                # print('Not Found')
    return cuilist


def get_file_list(filename):
    file_list = []
    with open(filename) as fp:
        while True:
            line = fp.readline()
            if not line :
                break
            if "list.txt" not in line:
                file_list.append(line)
    return file_list


def main():
    file_list = get_file_list("./Task1TrainSetGOLD200pipe/list.txt")
    print("Number of files : ", str(len(file_list)))
    P = []
    R = []
    F = []
    for fp in file_list:
        fp = re.sub(r'\n', '', fp)
        cui_list = get_CUI_to_find("./Task1TrainSetGOLD200pipe/" + fp)
        print("liste = ", cui_list)
        fp = re.sub(r'.pipe', '', fp)
        precision = compute_precision("./data/outputchunkssmall/" + fp + ".output", cui_list)
        recall = compute_recall("./data/outputchunkssmall/" + fp + ".output", cui_list)
        f1 = get_F1_score(precision, recall)
        P.append(precision)
        R.append(recall)
        F.append(f1)
    print("Precision : " + str(sum(P)/len(P)))
    print("Recall : " + str(sum(R)/len(R)))
    print("F1-Score : " + str(sum(F)/len(F)))


if __name__ == '__main__':
    main()
