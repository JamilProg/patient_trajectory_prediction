
from pymetamap import MetaMap
import csv, os, sys, time

if __name__ == "__main__":
    start_time = time.time()
    mm = MetaMap.get_instance('/home/jamilz/MetaMap/public_mm/bin/metamap18')
    myDict = {}
    dirchunks = "./data/chunkssmall/"
    diroutputchunks = "./data/outputchunkssmall/"
    counter = 0
    for file in os.listdir(dirchunks):
        filename = dirchunks+file
        liste_concepts = []
        lineNb=1
        with open(filename, 'r') as fd:
            print("File", filename, "opened! \nNow treating line: ", flush=True)
            for line in fd.readlines():
                print(lineNb, flush=True)
                if line not in myDict.keys():
                    # Match attempt
                    list_sent = []
                    list_sent.append(line)
                    concepts,error = mm.extract_concepts(list_sent,[1])
                    myDict[line] = concepts
                else:
                    concepts = myDict[line]
                    counter+=1
                # if counter % 100 == 0:
                #     print(counter)
                concepts_output = []
                for concept in concepts:
                    print(concept)
                    # concepts_output.append(list_terms)
                #liste_concepts.append(concepts_output)
                lineNb+=1
        # outputFile = diroutputchunks+file+".output"
        # print("Now saving data in", outputFile, flush=True)
        # with open(outputFile, 'w') as fd:
        #     for lineConcepts in liste_concepts:
        #         # fd.write(conceptListLine2text(lineConcepts)+'\n')
        #         resultline = ""
        #         for concepts in lineConcepts:
        #             for terms in concepts:
        #                 resultline += terms + " "
        #         fd.write(resultline+'\n')
        # fd.close()
    elapsed_time = time.time() - start_time
    print(elapsed_time, "seconds elapsed")
