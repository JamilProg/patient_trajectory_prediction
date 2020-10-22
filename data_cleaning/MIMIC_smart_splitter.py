#!/usr/bin/python
import os, sys


def splitDocument(sizeInMo):
    """Split the MIMIC III document for every 50 Mo (about) without cutting a note"""
    dirchunks = "./data/chunkssmall/"
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/chunkssmall'):
        os.makedirs('data/chunkssmall')
    if not os.path.exists('data/outputchunkssmall'):
        os.makedirs('data/outputchunkssmall')
    i = 1
    make_new_file = True
    outputFile = ""
    with open(sys.argv[1]) as fread:
        fread.readline() # avoid first line
        for line in fread.readlines():
            count_comma = line.count(',')
            count_quote = line.count('"')
            if count_comma >= 10 and count_quote >= 1:
                if make_new_file :
                    make_new_file = False
                    outputFile = dirchunks+str(i)+".csv"
            with open(outputFile, 'a') as fwrite:
                fwrite.write(line)
            if os.path.getsize(outputFile) > sizeInMo*1000000 and make_new_file is False:
                i += 1
                make_new_file = True


def main():
    """ Provides an argument : a path to the csv file (including the name of the csv) """
    if len(sys.argv) != 2:
        print("One argument is necessary : the path to the csv file")
        return -1
    splitDocument(50)


if __name__ == '__main__':
    main()
