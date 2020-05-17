#!/usr/bin/python
import sys
import re

ALLOWED_TYPES = ["antb", "bacs", "bodm", "chem", "chvf", "clnd", "enzy", "aapp", "elii", "hops", "irda", "imft", "horm", "phsu", "rcpt", "orga", "orgf", "ortf", "phsf", "moft", "acab", "anab", "comd", "cgab", "dsyn", "emod", "fndg", "inpo", "mobd", "neop", "tmco", "medd", "resd", "aggp", "amph", "anim", "arch", "bact", "bird", "famg", "fish", "fngs", "grup", "drdd", "euka", "blor", "vita", "chvs", "nnon", "inch", "orch", "patf", "sosy", "bmod", "ocdi", "anst", "biof", "lbtr", "npop", "phpr", "hcpp", "eehu", "clna", "celf", "genf", "menp", "bpoc", "bsoj", "bdsu", "bdsy", "cell", "celc", "emst", "ffas", "tisu", "humn", "mamm", "orgm", "podg", "popg", "prog", "rept", "vtbt", "virs", "hlca", "diap", "topp"]


def metamap_preprocessing(filename, outputfile):
    processed_file = open(outputfile, 'w')
    note_ended = True
    negative_CUI_regex = r'N C\d\d\d\d\d\d\d'
    CUI_regex = r'C\d\d\d\d\d\d\d'
    type_regex = r'\[(.*)\]'
    current_str_to_add = ""
    pattern = re.compile(r'^.*?,.*?,\d+,')
    with open(filename, 'r') as fp:
        while True:
            cleaned_line = ""
            line = fp.readline()
            if not line:
                break
            if line.startswith("Processing USER.tx.1: ") and note_ended == True and line.count(',') == 10:
                cleaned_line = line.replace("Processing USER.tx.1: ", "")
                # processed_file.write(cleaned_line)
                # Avoid null HADM ID
                if not re.match(pattern, cleaned_line) :
                    print(cleaned_line)
                    continue
                # If full admission is not computed
                if not cleaned_line.count('"')%2 == 0:
                    index_to_last_quote = cleaned_line.rfind("\"")
                    cleaned_line = cleaned_line[:index_to_last_quote+1]
                    current_str_to_add = cleaned_line
                    note_ended = False
            elif line.startswith("Processing USER.tx.1: ") and note_ended == False and line.count('"') == 1:
                cleaned_line = line.replace("Processing USER.tx.1: ", "")
                note_ended = True
                # processed_file.write('"\n')
                current_str_to_add += '"\n'
                processed_file.write(current_str_to_add)
                current_str_to_add = ""
            elif note_ended == False :
                r = re.search(CUI_regex, line)
                s = re.search(negative_CUI_regex, line)
                t = re.search(type_regex, line)
                type_list = []
                if t :
                    # Get semantic type
                    types = t.group(1)
                    types = re.sub(',', ' ', types)
                    type_list = types.split()
                if s and t:
                    # Negative CUI case
                    cui = s.group(0)
                    cui = re.sub(r' ', '', cui)
                    for tui in type_list:
                        if tui in ALLOWED_TYPES:
                            # processed_file.write(cui + " ")
                            current_str_to_add = current_str_to_add + cui + " "
                            break
                elif r and t:
                    # Positive CUI case
                    cui = r.group(0)
                    for tui in type_list:
                        if tui in ALLOWED_TYPES:
                            # processed_file.write(cui + " ")
                            current_str_to_add = current_str_to_add + cui + " "
                            break


if __name__ == '__main__':
    """ Provides an argument : a path to the metamap file """
    if len(sys.argv) != 2:
        print("One argument is necessary : the path to the metamap output file")
        exit()
    metamap_preprocessing(sys.argv[1], "metamap_output.csv")
