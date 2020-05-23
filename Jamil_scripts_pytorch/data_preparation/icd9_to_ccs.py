#!/usr/bin/python
import csv


def getCCS(icd_code):
    # Step 1 : get the dictionary
    reader = csv.reader(open('ICDToCCS.csv', 'r'))
    d = dict(reader)
    if (icd_code not in d.keys()):
        return
    else:
        # print("CODE FOUND:", d[icd_code])
        return d[icd_code]
