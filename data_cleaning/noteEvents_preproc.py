#!/usr/bin/python
import sys
import os
from utils_preproc import *


def shape_to_csv(inputfile, outputfile):
    """ Ensures that we have a CSV shape : we separate the text columns from
    the others columns with breaklines in the right places """
    processed_file = open(outputfile, 'w')
    count_comma = 0
    within_text = False
    with open(inputfile) as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            if count_comma >= 10 or within_text:
                if line.count("\"") == 0:
                    processed_file.write(line)
                    continue
            else:
                if line.count(",") == 0:
                    processed_file.write(line)
                    continue
            ## Some cases that could break the pipeline !
            # if "Resident Progress Note, CCU" in line : # one additional comma to count
            #     count_comma -= 1
            # elif "Code Blue, Cardiac Arrest, Death" in line : # two add. comma to count
            #     count_comma -= 2
            # ...
            ### START This part of code generalizes the previous example
            cntcm = line.count(',')
            cntqt = line.count('"')
            if cntqt != 0 and cntcm != 0 :
                if cntcm != 10 :
                    if cntqt > 1 :
                        groups = line.split('"')
                        test = '"'.join(groups[:cntqt]), '"'.join(groups[cntqt:])
                        if test[0].count(',') != 10 :
                            count_comma -= test[0].count(',') - 10
            ### END This part of code generalizes the previous example
            index = -1
            for c in line:
                index += 1
                if c == ',' and within_text == False:
                    count_comma += 1
                if c == '\"':
                    if count_comma >= 10:
                        count_comma = 0
                        within_text = True
                        line = line[:index+1] + '\n' + line[index+1:]
                        index += 1
                    elif within_text:
                        within_text = False
                        line = line[:index+1] + '\n' + line[index+1:]
                        line = line[:index] + '\n' + line[index:]
                        index += 2
            processed_file.write(line)
    processed_file.close()


def remove_extra_commas(inputfile, outputfile):
    """ Toss off commas that are within the Description field """
    processed_file = open(outputfile, 'w')
    with open(inputfile) as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            commas = line.count(',')
            if line.count("\"") > 0 :
                if (commas == 0 or commas == 10) :
                    processed_file.write(line)
                if commas > 10 :
                    # remove_extra_commas
                    cleaned_line = re.sub(r'"[^"]*"', lambda m: m.group(0).replace(',', ''), line)
                    processed_file.write(cleaned_line)
            else :
                processed_file.write(line)
    processed_file.close()


def toss_off_rare_words(inputfile, outputfile, word_dict):
    """ Toss off words that occur less than 5 times in the corpus """
    processed_file = open(outputfile, 'w')
    a_subset = {key: value for key, value in word_dict.items() if value < 5}
    print("Tossing off rare words.\nSize of words with less than 5 frequency:", len(a_subset))
    with open(inputfile) as fp:
        while True:
            line = fp.readline()
            if line == "\n" or "," in line or "\"" in line:
                processed_file.write(line)
                continue
            if not line:
                break
            word_list = word_tokenize(line)
            for w in word_list :
                if w in a_subset.keys():
                    re.sub(w, '', line)
            processed_file.write(line)
    processed_file.close()
    # If you want to see the tossed off words, uncomment the next line
    # print(a_subset.items())


def preprocess_enumerations(inputfile, outputfile):
    """ Recognize if a numerated list is a list of paragraph or a list of elements
    We do so by implementing thresholds : max # char and avg # char over all lines the list
    In the meantime, we also remove the digits. and # part of the lines
    """
    processed_file = open(outputfile, 'w')
    regex_start_enum = re.compile(r'^[1-9][0-9]?\. +|^#')
    start_of_enum = False
    current_list = ""
    len_list = []
    with open(inputfile) as fp:
        while True:
            line = fp.readline()
            if not line:
                if start_of_enum:
                    # write the last
                    if max(len_list) < 300:
                        if sum(len_list) / len(len_list) < 250:
                            current_list = re.sub(r'\n', ' ', current_list)
                    else:
                        if sum(len_list) / len(len_list) < 150:
                            current_list = re.sub(r'\n', ' ', current_list)
                    processed_file.write(current_list)
                    start_of_enum = False # do not forget to disable it
                break
            if regex_start_enum.match(line) and line.count('"')==0:
                cleaned_line = re.sub(r'^[1-9][0-9]?\. +|^#+ *', '', line)
                current_list += cleaned_line
                len_list.append(len(cleaned_line))
                start_of_enum = True
            else:
                if start_of_enum:
                    # Uncomment the three following lines to see the max and avg values for each list
                    # print(current_list)
                    # print("Average : " + str(sum(len_list)/len(len_list)))
                    # print("Max : " + str(max(len_list)))
                    # Now that we have the needed info of the list, we clean it
                    if max(len_list) < 300:
                        if sum(len_list)/len(len_list) < 250:
                            current_list = re.sub(r'\n', ' ', current_list)
                    else:
                        if sum(len_list) / len(len_list) < 150:
                            current_list = re.sub(r'\n', ' ', current_list)
                    processed_file.write(current_list)
                    current_list = ""
                    start_of_enum = False
                    len_list = []

                if line.count('"')!= 0:
                    line = '\n' + line
                processed_file.write(line)
    processed_file.close()


def clean_useless_words(inputfile, outputfile):
    """
    1) This procedure removes the useless part of the first line of the text : "Admission Date: " "Discharge Date: "
    2) It removes the useless words (DATE OF BIRTH, SERVICE, SEX, ADDENDUM)
    3) At the end of many texts, there are some useless infos such as (JOB#, D:, T:, Dictated by:,...)
    4) Removes the days of a week
    """
    processed_file = open(outputfile, 'w')
    with open(inputfile) as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            # 1st task
            cleaned_line = re.sub(r'admission date:.*', '', line)
            # 2nd task
            cleaned_line = re.sub(r'sex *?: *[mf]?[ \n]', '', cleaned_line)
            cleaned_line = re.sub(r'date of birth *?:', '', cleaned_line)
            cleaned_line = re.sub(r'service *?: *?.*?\n$', '', cleaned_line)
            cleaned_line = re.sub(r'addendum *?:?', '', cleaned_line)
            cleaned_line = re.sub(r'medquist36', '', cleaned_line)
            cleaned_line = re.sub(r'm\.d\.', '', cleaned_line)
            cleaned_line = re.sub(r'\Wmd\W|^md\W', '', cleaned_line)
            # 3rd task
            cleaned_line = re.sub(r'^dictated *?by *?: *', '', cleaned_line)
            cleaned_line = re.sub(r'^completed *?by *?: *', '', cleaned_line)
            cleaned_line = re.sub(r'^cc *?(by)? *?: *', '', cleaned_line)
            cleaned_line = re.sub(r'^d *?: *(\d\d:\d\d)?', '', cleaned_line)
            cleaned_line = re.sub(r'^t *?: *(\d\d:\d\d)?', '', cleaned_line)
            cleaned_line = re.sub(r'phone *?: *', '', cleaned_line)
            cleaned_line = re.sub(r'provider *?: *', '', cleaned_line)
            cleaned_line = re.sub(r'date/time *?: *', '', cleaned_line)
            cleaned_line = re.sub(r'^job# *?: *', '', cleaned_line)
            # 4th task
            cleaned_line = re.sub(r'monday', '', cleaned_line)
            cleaned_line = re.sub(r'tuesday', '', cleaned_line)
            cleaned_line = re.sub(r'wednesday', '', cleaned_line)
            cleaned_line = re.sub(r'thursday', '', cleaned_line)
            cleaned_line = re.sub(r'friday', '', cleaned_line)
            cleaned_line = re.sub(r'saturday', '', cleaned_line)
            cleaned_line = re.sub(r'sunday', '', cleaned_line)
            processed_file.write(cleaned_line)
    processed_file.close()


def special_char_remover(inputfile, outputfile):
    """ Removes special chars : they are irrelevant - word2vec doesn't like that """
    processed_file = open(outputfile, 'w')
    with open(inputfile) as fp:
        while True:
            line = fp.readline()
            if not line:
                break

            # Avoid removing special chars in the IDs row
            count_comma = line.count(',')
            count_quote = line.count('"')
            if count_comma >= 10 and count_quote >= 1:
                processed_file.write(line)
                continue

            cleaned_line = re.sub(r'[*<>!?#.^;$&~_/\\]', '', line)
            cleaned_line = re.sub(r'[-+=():,\']', ' ', cleaned_line)
            cleaned_line = re.sub(r'\[|\]', ' ', cleaned_line)
            cleaned_line = re.sub(r'\w%', ' percent', cleaned_line)
            cleaned_line = re.sub(r'%', 'percent', cleaned_line)
            processed_file.write(cleaned_line)
    processed_file.close()


def time_remover(inputfile, outputfile):
    """ Remove time present in the file
    HH:MM information are completely irrelevant in the eyes of the machine because it does not have any influence
    on patient's health, future diagnoses, and so on. So we remove them.
    """
    processed_file = open(outputfile, 'w')
    with open(inputfile) as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            if line.count("\"") > 0:
                processed_file.write(line)
                continue
            cleaned_line = re.sub(r'\d?\d:\d\d *?((am|pm)\W)?', '', line)
            cleaned_line = re.sub(r'\d?\d:\d\d:\d\d *?((am|pm)\W)?', '', cleaned_line)
            processed_file.write(cleaned_line)
    processed_file.close()


def repetitive_number_parentheses(inputfile, outputfile):
    """ Doctors write a lot numbers in letters followed by the actual number within parenthesis, like that :
    << He should take one (1) at bedtime and two (2) in the morning. >>
    We remove the parenthesis part
    """
    processed_file = open(outputfile, 'w')
    with open(inputfile) as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            cleaned_line = re.sub(r'\(\d\) ?', '', line)
            processed_file.write(cleaned_line)
    processed_file.close()


def numbers_to_text(inputfile, outputfile):
    """ Transform numbers to their textual version - word2vec performs better with textual info
        WARNING : do not convert IDs into textual, only do that to numbers inside the TEXT cells.
    """
    processed_file = open(outputfile, 'w')
    with open(inputfile) as fp:
        while True:
            line = fp.readline()
            if not line:
                break

            # Avoid transforming IDs to letters (it doesn't make sense at all)
            count_comma = line.count(',')
            count_quote = line.count('"')
            if count_comma >= 10 and count_quote >= 1:
                processed_file.write(line)
                continue

            # if \d+\.\d+ is found, transform . to [space]point[space]
            # (and if there are zeros after the dot, replace them by "zero ")
            cleaned_line = re.sub(r'((\d|)*)\.00(\d+)', r' \1 point zero zero \3 ', line)
            cleaned_line = re.sub(r'((\d|)*)\.0(\d+)', r' \1 point zero \3 ', cleaned_line)
            cleaned_line = re.sub(r'((\d|)*)\.(\d+)', r' \1 point \3 ', cleaned_line)
            # for all digits found, replace it by the text form (sub)
            cleaned_line = re.sub(r'([1-9]\d*|0)', lambda x: num_to_words(x.group()), cleaned_line)
            processed_file.write(cleaned_line)
    processed_file.close()


def spaces_remover(inputfile, outputfile):
    """ This procedure does two things :
    1) It removes all spaces that are starting a paragraph (a line)
    2) It replaces every "more than 1 space in a row" by 1 space
    """
    processed_file = open(outputfile, 'w')
    with open(inputfile) as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            # If paragraph starts with spaces, remove all of them
            cleaned_line = re.sub(r'^ +', '', line)
            # If more than two spaces within paragraph : leave it with two spaces
            cleaned_line = re.sub(r' +', ' ', cleaned_line)
            processed_file.write(cleaned_line)
    processed_file.close()


def doctor_quotes_remover(inputfile, outputfile):
    """ Doctor quotes mark "" "" are removed """
    processed_file = open(outputfile, 'w')
    with open(inputfile) as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            cleaned_line = re.sub('""', '', line)
            processed_file.write(cleaned_line)
    processed_file.close()


def anonimization_remover(inputfile, outputfile):
    """ Anonimization mark '[** **]' and its content are removed"""
    processed_file = open(outputfile, 'w')
    with open(inputfile) as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            cleaned_line = re.sub(r'\[\*\*.*?\*\*\]', '', line)
            processed_file.write(cleaned_line)
    processed_file.close()


def lower_all_text(inputfile, outputfile):
    """ Every letter in the text becomes lowercase """
    processed_file = open(outputfile, 'w')
    with open(inputfile) as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            cleaned_line = line.lower()
            processed_file.write(cleaned_line)
    processed_file.close()


def paragraph_finder(inputfile, outputfile):
    """ This function does the following :
    1) If we find a case where we have "{anything}\n([NUMBER].|#){anything}", we recognize that there is a new paragraph
    2) If we find two or more occurrences of '\n' in a row, we keep at least one occurrence (new paragraph)
    3) Otherwise, if the two previous rules don't apply, we replace '\n' by a ' ' as we think we are within a paragraph.
    """
    # REGEX : if the string contains a number followed by a dot, it is the start of a new paragraph
    reg_exp_new_line = re.compile(r'^[1-9][0-9]?\. +|^#')
    # REGEX : if a line match this, consider this line as an empty line
    bad_line_re = re.compile(r'^[,.]* *\n$')
    # BOOL : indicate if we had an empty line before (in this case, the next line starts a new paragraph for sure)
    previous_was_empty = 0
    # Prepare the output file
    processed_file = open(outputfile, 'w')
    paragraph = ""
    with open(inputfile) as fp:
        line = fp.readline()
        paragraph += line
        while line:
            line = fp.readline()
            # if line != "\n" and line != ".\n":
            if line.count("\"")>0 :
                if line.count("\"") == 1:
                    line = '\n'+line
                    processed_file.write(paragraph)
                    paragraph = line
                    continue
            if not bad_line_re.match(line):
                if previous_was_empty:
                    # Save paragraph, now we have a new one
                    paragraph += "\n"
                    processed_file.write(paragraph)
                    line = replace_breakline_by_space(line, get_next_line_without_moving(fp))
                    paragraph = line
                elif reg_exp_new_line.match(line):
                    # Save paragraph, now we have a new one
                    paragraph += "\n"
                    processed_file.write(paragraph)
                    line = replace_breakline_by_space(line, get_next_line_without_moving(fp))
                    paragraph = line
                else:
                    line = replace_breakline_by_space(line, get_next_line_without_moving(fp))
                    paragraph += line
                previous_was_empty = 0
            else:
                previous_was_empty = 1
    processed_file.close()


def main():
    """ Provides an argument : a path to the csv file (including the name of the csv) """
    if len(sys.argv) != 2:
        print("One argument is necessary : the path to the NOTEEVENTS.csv file")
        return -1
    # New folder in which we will save the output
    # outpath = "out_results/"
    # os.mkdir(outpath)
    outpath = ""
    # Run preprocessing over the input
    anonimization_remover(sys.argv[1], outpath+'out_noanonim.csv')
    doctor_quotes_remover(outpath+'out_noanonim.csv', outpath+'out_nodocquotes.csv')
    os.remove(outpath+'out_noanonim.csv')
    shape_to_csv(outpath+'out_nodocquotes.csv', outpath+'out_csvshape.csv')
    os.remove(outpath+'out_nodocquotes.csv')
    remove_extra_commas(outpath+'out_csvshape.csv', outpath+'out_noextracommas.csv')
    os.remove(outpath+'out_csvshape.csv')
    lower_all_text(outpath+'out_noextracommas.csv', outpath+'out_lower.csv')
    os.remove(outpath+'out_noextracommas.csv')
    clean_useless_words(outpath+'out_lower.csv', outpath+'out_nobadwords.csv')
    os.remove(outpath+'out_lower.csv')
    time_remover(outpath+'out_nobadwords.csv', outpath+'out_notime.csv')
    os.remove(outpath+'out_nobadwords.csv')
    repetitive_number_parentheses(outpath+'out_notime.csv', outpath+'out_noparentheses.csv')
    os.remove(outpath+'out_notime.csv')
    spaces_remover(outpath+'out_noparentheses.csv', outpath+'out_nospaces.csv')
    os.remove(outpath+'out_noparentheses.csv')
    paragraph_finder(outpath+'out_nospaces.csv', outpath+'out_paragraphs.csv')
    os.remove(outpath+'out_nospaces.csv')
    preprocess_enumerations(outpath+'out_paragraphs.csv', outpath+'out_enum.csv')
    os.remove(outpath+'out_paragraphs.csv')
    numbers_to_text(outpath+'out_enum.csv', outpath+'out_nonumbers.csv')
    os.remove(outpath+'out_enum.csv')
    special_char_remover(outpath+'out_nonumbers.csv', outpath+'out_nospecchar.csv')
    os.remove(outpath+'out_nonumbers.csv')
    spaces_remover(outpath+'out_nospecchar.csv', outpath+'out_nospaces2.csv')
    os.remove(outpath+'out_nospecchar.csv')
    word_dico = get_vocabulary(outpath+'out_nospaces2.csv')
    toss_off_rare_words(outpath+'out_nospaces2.csv', outpath+'out_norare.csv', word_dico)
    os.remove(outpath+'out_nospaces2.csv')
    spaces_remover(outpath+'out_norare.csv', outpath+'output.csv')
    os.remove(outpath+'out_norare.csv')

    # Those lines show how the curve changed after we started to consider enumerations as a single paragraph
    # get_paragraph_distribution('out_paragraphs.csv')
    # get_paragraph_distribution('out_enum.csv')
    # get_paragraph_distribution('out_nonumbers.csv')


if __name__ == '__main__':
    main()
