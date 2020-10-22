#!/usr/bin/python
import re
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize


# Dico for references
ones = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
        "6": "six", "7": "seven", "8": "eight", "9": "nine"}
afterones = {"10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen",
             "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen"}
tens = {"2": "twenty", "3": "thirty", "4": "fourty", "5": "fifty",
        "6": "sixty", "7": "seventy", "8": "eighty", "9": "ninety"}
grand = {0: " billion ", 1: " million ", 2: " thousand ", 3: ""}


nltk.download('punkt')


def get_next_line_without_moving(f):
    pos = f.tell()
    line = f.readline()
    line = f.readline()
    f.seek(pos)
    return line


def get_vocabulary(inputfile):
    """ This procedure takes a MIMIC NoteEvents file and returns a dictionary
    which contains words and their corresponding count """
    # Ignore first line (columns title)
    # If comma in the line, ignore it as it is NOT text
    # Otherwise, take the line, and foreach word in line, if word in dict.keys(), count++, otherwise new words
    word_dict = dict()
    with open(inputfile) as fp:
        # Ignore first line
        line = fp.readline()
        while True:
            line = fp.readline()
            if line == "\n" or "," in line or "\"" in line:
                continue
            if not line:
                break
            word_list = word_tokenize(line)
            for w in word_list :
                if w in word_dict.keys():
                    word_dict[w] += 1
                else:
                    word_dict[w] = 1
    print("Vocabulary size:", len(word_dict))
    return word_dict


def show_histogram(distribution, n_bins, title):
    plt.style.use('ggplot')
    plt.title(title)
    plt.hist(distribution, bins=n_bins)
    plt.show()


def get_paragraph_distribution(inputfile):
    """ Displays the number of paragraph in the file for each size of character"""
    # Array saving the length of paragraphs
    par_lengths = []
    with open(inputfile) as fp:
        while True:
            line = fp.readline()
            if line == "\n":
                continue
            if not line:
                break
            par_lengths.append(len(line))
    # Now we display the histograms
    show_histogram(par_lengths, max(par_lengths), 'Number of paragraph with respect to its size')


def replace_breakline_by_space(given_line, next_line):
    """ Replaces '\n' by ' ' at the end of the given line if exists
    This function is called by paragraphFinder
    """
    if len(given_line) == 0:
        return given_line
    if given_line.count('"') > 0 :
        return given_line
    if next_line.count('"') > 0 :
        return given_line
    if given_line[len(given_line)-1] != '\n':
        return given_line
    given_line = given_line.replace(given_line[len(given_line)-1], ' ')
    return given_line


def three_dig_to_words(val):
    """ Function converting number to words of 3 digit
    Code from Barath Kumar
    Link : https://stackoverflow.com/questions/15598083/python-convert-numbers-to-words
    """
    if val != "000":
        ans = ""
        if val[0] in ones:
            ans = ans + ones[val[0]] + " hundred "
        if val[1:] in afterones:
            ans = ans + afterones[val[1:]] + " "
        elif val[1] in tens:
            ans = ans + tens[val[1]] + " "
        if val[2] in ones and val[1:] not in afterones:
            ans = ans + ones[val[2]]
        return ans


def num_to_words(value):
    """ This function takes an integer as an input, and outputs its text version
    Works with integer from 0 to 999 999 999 999.
    """
    # Padding with zeros
    pad = 12 - len(str(value))
    padded = "0" * pad + str(value)

    # Exception case
    if padded == "000000000000":
        return "zero"

    # Preparing the values before computation
    result = ""
    number_groups = [padded[0:3], padded[3:6], padded[6:9], padded[9:12]]

    for key, val in enumerate(number_groups):
        if val != "000":
            result = result + three_dig_to_words(val) + grand[key]

    result = re.sub(r'(^ *| *$)', ' ', result)
    return result
