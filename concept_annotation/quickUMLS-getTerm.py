from quickumls import QuickUMLS
import csv, os, sys, time, re


if __name__ == "__main__":
    # start_time = time.time()

    THRESHOLD = 0.7
    matcher = QuickUMLS(quickumls_fp='./QuickUMLS', overlapping_criteria='score', threshold=THRESHOLD, similarity_name='cosine', window=5)
    myDict = {}
    dirchunks = "./data/chunkssmall/"
    diroutputchunks = "./data/outputchunkssmall/"
    for file in os.listdir(dirchunks):
        filename = dirchunks+file
        liste_concepts = []
        lineNb=1
        with open(filename, 'r') as fd:
            print("File", filename, "opened! \nNow treating line: ", flush=True)
            # Preparing outputfile
            outputFile = diroutputchunks+file+".output"
            fw = open(outputFile, 'w')
            for line in fd.readlines():
                # Keep IDs and non-text information
                count_comma = line.count(',')
                count_quote = line.count('"')
                if count_comma >= 10 and count_quote >= 1:
                    # New clinical note
                    fw.write(line)
                    continue
                print(lineNb, flush=True)
                if line not in myDict.keys():
                    matches  = matcher.match(line, best_match=True, ignore_syntax=False)
                    # print(matches)
                    myDict[line] = matches
                else:
                    matches = myDict[line]
                concepts_output = []
                for phrase_candidate in matches:
                    # Find max
                    max=0
                    # print("PC :", phrase_candidate)
                    for candidate in phrase_candidate:
                        if candidate['similarity']>max:
                            max=candidate['similarity']
                    # Get preferred terms for that max
                    list_to_write = []
                    for candidate in phrase_candidate:
                        if candidate['similarity']==max:
                            # print("C : ", candidate)
                            if candidate['term'] not in list_to_write:
                                list_to_write.append(candidate['term'])
                    concepts_output.append(list_to_write)
                    resultline = ""
                    for concepts in concepts_output:
                        for terms in concepts:
                            terms = re.sub(r' ', '', terms)
                            resultline += terms + " "
                    fw.write(resultline+'\n')
                    concepts_output = []
                    lineNb+=1
                if count_quote >= 1 :
                    # End of clinical note
                    fw.write('"\n')
            fw.close()

    # elapsed_time = time.time() - start_time
    # print(elapsed_time, "seconds elapsed")
