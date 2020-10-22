from quickumls import QuickUMLS
import csv, os, sys, time, re    


if __name__ == "__main__":
    # start_time = time.time()

    THRESHOLD = 0.7
    matcher = QuickUMLS(quickumls_fp='./QuickUMLS', overlapping_criteria='score', threshold=THRESHOLD, similarity_name='cosine', window=5)
    # myDict = {}
    dirchunks = "./data/chunkssmall/"
    diroutputchunks = "./data/outputchunkssmall/"
    # list_cui = []
    # list_terms = []
    for file in os.listdir(dirchunks):
        filename = dirchunks+file
        # liste_concepts = []
        lineNb=1
        list_cui = []
        list_terms = []
        with open(filename, 'r') as fd:
            print("File", filename, "opened! \nNow treating line: ", flush=True)
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
                print(lineNb, flush=True)
                # if line not in myDict.keys():
                    # matches  = matcher.match(line, best_match=True, ignore_syntax=False)
                    # print(matches)
                    # myDict[line] = matches
                # else:
                    # matches = myDict[line]
                matches = matcher.match(line, best_match=True, ignore_syntax=False)
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

        # outputFile = diroutputchunks+file+".output"
        # print("Now saving data in", outputFile, flush=True)
        # with open(outputFile, 'w') as fd:
        #     for lineConcepts in liste_concepts:
        #         resultline = ""
        #         for concepts in lineConcepts:
        #             for terms in concepts:
        #                 terms = re.sub(r' ', '', terms)
        #                 resultline += terms + " "
        #         fd.write(resultline+'\n')
        # fd.close()

    # elapsed_time = time.time() - start_time
    # print(elapsed_time, "seconds elapsed")
