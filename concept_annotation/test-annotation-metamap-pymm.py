#!/usr/bin/env python

from pymm import Metamap
import csv, os, sys, time
from os.path import exists

BATCH_SIZE = 50
METAMAP_PATH = "/home/jamilz/MetaMap/public_mm/bin/metamap18"


def read_lines(file_name, fast_forward_to, batch_size):
    sentences = list()
    with open(file_name, 'r') as fp:
        for i in range(fast_forward_to):
            fp.readline()

        for idx, line in enumerate(fp):
            sentences.append(line)
            if (idx+1) % batch_size == 0:
                yield sentences
                sentences.clear()


if __name__ == "__main__":
    start_time = time.time()
    # mm = Metamap('/home/jamilz/MetaMap/public_mm/bin/metamap18')
    # myDict = {}
    dirchunks = "./data/chunkssmall/"
    diroutputchunks = "./data/outputchunkssmall/"
    mm = Metamap(METAMAP_PATH)
    print("MetaMap running :", mm.is_alive())
    # assert mm.is_alive()
    last_checkpoint = BATCH_SIZE
    try:
        for i, sentences in enumerate(read_lines(dirchunks + "00098-016139-DISCHARGE_SUMMARY.txt", last_checkpoint, BATCH_SIZE)):
            timeout = 0.33*BATCH_SIZE
            try_again = False
            try:
                mmos = mm.parse(sentences, timeout=timeout)
            except MetamapStuck:
                # Try with larger timeout
                print ("Metamap Stuck !!!; trying with larger timeout")
                try_again = True
            except:
                print ("Exception in mm; skipping the batch")
                traceback.print_exc(file=sys.stdout)
                continue

            if try_again:
                timeout = BATCH_SIZE*2
                try:
                    mmos = mm.parse(sentences, timeout=timeout)
                except MetamapStuck:
                    # Again stuck; Ignore this batch
                    print ("Metamap Stuck again !!!; ignoring the batch")
                    continue
                except:
                    print ("Exception in mm; skipping the batch")
                    traceback.print_exc(file=sys.stdout)
                    continue

            for idx, mmo in enumerate(mmos):
                for jdx, concept in enumerate(mmo):
                    # save(sentences[idx], concept)
                    print (concept.cui, concept.score, concept.matched)
                    print (concept.semtypes, concept.ismapping)

            curr_checkpoint = (i+1)*BATCH_SIZE + last_checkpoint
            # record_checkpoint(curr_checkpoint)
    finally:
        mm.close()
    # counter = 0
    # for file in os.listdir(dirchunks):
    #     filename = dirchunks+file
    #     # liste_concepts = []
    #     lineNb=1
    #     with open(filename, 'r') as fd:
    #         print("File", filename, "opened! \nNow treating line: ", flush=True)
    #         for line in fd.readlines():
    #             print(lineNb, flush=True)
    #             if line not in myDict.keys():
    #                 # Match attempt
    #                 list_sent = []
    #                 list_sent.append(line)
    #                 concepts,error = mm.extract_concepts(list_sent,[1])
    #                 myDict[line] = concepts
    #             else:
    #                 concepts = myDict[line]
    #                 # counter+=1
    #             # if counter % 100 == 0:
    #             #     print(counter)
    #             concepts_output = []
    #             for concept in concepts:
    #                 print(concept)
    #                 # concepts_output.append(list_terms)
    #             #liste_concepts.append(concepts_output)
    #             lineNb+=1
    #     # outputFile = diroutputchunks+file+".output"
    #     # print("Now saving data in", outputFile, flush=True)
    #     # with open(outputFile, 'w') as fd:
    #     #     for lineConcepts in liste_concepts:
    #     #         # fd.write(conceptListLine2text(lineConcepts)+'\n')
    #     #         resultline = ""
    #     #         for concepts in lineConcepts:
    #     #             for terms in concepts:
    #     #                 resultline += terms + " "
    #     #         fd.write(resultline+'\n')
    #     # fd.close()
    elapsed_time = time.time() - start_time
    print(elapsed_time, "seconds elapsed")
