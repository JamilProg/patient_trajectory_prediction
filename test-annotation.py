
from quickumls import QuickUMLS
import csv, os, sys, time

# def topRankedConcepts(list_concepts):
#     "returns the list of preferred terms for max ranked concepts"
#     #Find max
#     max=0
#     for candidate in list_concepts:
#         if candidate['similarity']>max:
#             max=candidate['similarity']
#
#     #get preferred terms for that max
#     list_terms = []
#     for candidate in list_concepts:
#         if candidate['similarity']==max:
#             if candidate['term'] not in list_terms:
#                 list_terms.append(candidate['term'])
#     return list_terms
#
#
# def text2concepts(text):
#     "take one line, returns all top ranked concepts contained in the text (preferred term only)"
#     global matcher
#     global counter
#     global myDict
#     #print("debut matching")
#     if text not in myDict.keys():
#         matches  = matcher.match(text, best_match=True, ignore_syntax=False)
#         myDict[text] = matches
#     else:
#         matches = myDict[text]
#         counter+=1
#
#     if counter%100 == 0:
#       print(counter)
#
#     list_concepts = []
#     for phrase_candidate in matches:
#         list_concepts.append(topRankedConcepts(phrase_candidate))
#
#     return list_concepts
#
#     #for phrase_candidate in matches:
#     #    print(phrase_candidate[0]['term'])
#     #    for candidate_concept in phrase_candidate:
#     #            print(candidate_concept['cui'])
#
# def file2listOfConcepts(filename):
#     lconcepts = []
#     lineNb=1
#     with open(filename, 'r') as fd:
#         print("File", filename, "opened! \nNow treating line: ", flush=True)
#         for line in fd.readlines():
#             # print(lineNb, flush=True)
#             lconcepts.append(text2concepts(line))
#             lineNb+=1
#     return lconcepts
#
# def conceptListLine2text(list_concepts):
#     "for a list containing the concepts of a line, turns it into a string"
#     resultline = ""
#     for concepts in list_concepts:
#         for terms in concepts:
#             resultline += terms + " "
#     return resultline
#
# def list2outputFile(list_concepts,outputFile):
#     "writes the output as text in a file"
#     print("Now saving data in", outputFile, flush=True)
#     with open(outputFile, 'w') as fd:
#         for lineConcepts in list_concepts:
#             fd.write(conceptListLine2text(lineConcepts)+'\n')
#     fd.close()

if __name__ == "__main__":
    start_time = time.time()
    # text = "The ulna has dislocated posteriorly from the trochlea of the humerus."
    # text = "brain tumor is benign. The patient is cured. "
    # text1 = "The brain tumor is benign. The Patient is in the emergency service. "
    # text2 = "patient brain flu. Very painful."

    # fichiertest = "../data/test-small-file.csv"

    # listeConcepts = file2listOfConcepts(fichiertest)
    # print(listeConcepts)
    # print(conceptListLine2text(listeConcepts[2]))

    # list2outputFile(listeConcepts, "../data/outputtest")

    THRESHOLD = 0.7
    matcher = QuickUMLS(quickumls_fp='./QuickUMLS', overlapping_criteria='score', threshold=THRESHOLD, similarity_name='cosine', window=5)
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
                # liste_concepts.append(text2concepts(line))
                if line not in myDict.keys():
                    matches  = matcher.match(text, best_match=True, ignore_syntax=False)
                    myDict[line] = matches
                else:
                    matches = myDict[line]
                    counter+=1
                if counter%100 == 0:
                    print(counter)
                concepts_output = []
                for phrase_candidate in matches:
                    #concepts_output.append(topRankedConcepts(phrase_candidate))
                    #Find max
                    max=0
                    for candidate in phrase_candidate:
                        if candidate['similarity']>max:
                            max=candidate['similarity']

                    #get preferred terms for that max
                    list_terms = []
                    for candidate in phrase_candidate:
                        if candidate['similarity']==max:
                            if candidate['term'] not in list_terms:
                                list_terms.append(candidate['term'])
                    concepts_output.append(list_terms)
                liste_concepts.append(concepts_output)
                lineNb+=1
        # list2outputFile(liste_concepts, diroutputchunks+file+".output")
        outputFile = diroutputchunks+file+".output"
        print("Now saving data in", outputFile, flush=True)
        with open(outputFile, 'w') as fd:
            for lineConcepts in liste_concepts:
                # fd.write(conceptListLine2text(lineConcepts)+'\n')
                resultline = ""
                for concepts in lineConcepts:
                    for terms in concepts:
                        resultline += terms + " "
                fd.write(resultline+'\n')
        fd.close()
    elapsed_time = time.time() - start_time
    print(elapsed_time, "seconds elapsed")
