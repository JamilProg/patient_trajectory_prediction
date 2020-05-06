#!/usr/bin/python
import sys
import re


def checkScript():
    """ Outputs lines which contains comma and/or quote
    It is here to monitor (and check) if the preprocessed file is still
    a well-built csv file
    """
    with open(sys.argv[1]) as fread:
        while True:
            line = fread.readline()
            if not line:
                break
            count_comma = line.count(',')
            count_quote = line.count('"')
            # if count_quote != 0:
            #     print(count_quote, line)
            if count_comma != 0 and count_comma != 10 :
                print(count_comma, line)
            # if count_quote != 0  :
            #     if count_comma != 0 and count_comma != 10 :
            #         print(count_comma, line)
            # cntcm = line.count(',')
            # cntqt = line.count('"')
            # if cntqt != 0 and cntcm != 0 :
            #     if cntcm != 10 :
            #         if cntqt > 2 :
            #             groups = line.split('"')
            #             test = '"'.join(groups[:cntqt]), '"'.join(groups[cntqt:])
            #             if test[0].count(',') == 10 and count_comma != 10:
            #                 print(count_comma, line)
            #                 return


def main():
    """ Provides an argument : a path to the csv file (including the name of the csv) """
    if len(sys.argv) != 2:
        print("One argument is necessary : the path to the csv file")
        return -1
    checkScript()


if __name__ == '__main__':
    main()
