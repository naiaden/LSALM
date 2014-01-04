#! /usr/bin/python

import sys
import getopt
import time
import os.path
from subprocess import call

verbosity = int(0)
evaluateonly = False

dictionaryOutputFile = ''
corpusOutputFile = ''
evaluationOutputFile = ''

vocabularyInputFile = ''

countOutputDirectory = './counts'

order = 1

nrInputFiles = int(0)
nrTrainInstances = int(0)
dictionary = {}
dictionaryCounts = []
inputFiles = []
generateCountsFor = []

def printHelp():
    print("-h                 print this help and exit")
    print("-o, --order n      order of the n-gram language model (currently only n=1 is implemented)")
    print("                   order can also be n, to only create sentence histories (each line contains n words)")
    print("-m, --mfile f      save the corpus file in f (in matrix market format)")
    print("-d, --dfile f      save the dictionary in f")
    print("-e, --efile f      save the dictionary as an evaluation file")
    print("-E, --evaluateonly only save the evaluation file")
    print("-V, --vocabulary f read the vocabulary from f. Each line contains one word")
    print("-l, --files f      read tokenised input files from f. Each file is processed into counts")
    print("-c, --counts f     read count files from SRILM from f")
    print("-C, --countsdir d  save the count files generated with -l in directory d (default=./counts)")
    print("-v, --verbose n    set the verbosity level (default=0)")

def countTrainInstances():
    instances = 0
    for inputFile in inputFiles:
        instances += sum(1 for line in open(inputFile)) 
    return instances

def createOpenDictionary():
    instances = 0
    for inputFile in inputFiles:
        with open(inputFile, 'r') as f:
            for line in f:
                instances += 1

                words = line.rstrip().split()
                for word in words:
                    if not word in dictionary:
                        dictionary[word] = len(dictionary)+1
    return instances

def readVocabularyAsDictionary():
    with open(vocabularyInputFile, 'r') as f:
        for line in f:
            line = line.rstrip()
            if not line in dictionary:
                dictionary[line] = len(dictionary)+1

def condPrint(level,text):
    if(level <= verbosity):
        print(text)

def createCountsFor(tokFile):
    (d, fn) = os.path.split(tokFile)
    (f, e) = os.path.splitext(fn)

    countOutputFile = "%s/%s.%s" % (countOutputDirectory, f, "count")
    call(["ngram-count", "-text", tokFile, "-order", str(order), "-no-sos", "-no-eos", "-write", countOutputFile, "-gt1min", "0", "-gt2min", "0", "-gt3min", "0", "-gt4min", "0", "-gt5min", "0"])
    inputFiles.append(countOutputFile)

def createEvaluationFile():
    with open(evaluationOutputFile, 'w') as eof:
        for inputFile in inputFiles:
            with open(inputFile, 'r') as f:
                for line in f:
                    ngram = line.split('\t')[0]
                    if len(ngram.split()) == order:
                        eof.write("%s\n" % ngram)

def createSentenceHistories():
    with open(evaluationOutputFile, 'w') as eof:
        for inputFile in generateCountsFor:
            with open(inputFile, 'r') as f:
                for line in f:
                    words = line.rstrip().split()
                    for i in range(len(words)):
                        eof.write("%s\n" % ' '.join(words[0:1+i]))
                

try:
    opts, args = getopt.getopt(sys.argv[1:], 'ho:m:d:e:EV:l:c:C:v:', ['help', 'order=', 'mfile=', 'dfile=', 'efile=', 'evaluateonly', 'vocabulary=', 'files=', 'counts=', 'countsdir=', 'verbose=' ])
except getopt.GetoptError:
    printHelp()
    sys.exit(2)

for (opt, arg) in opts:
    if opt == '-h':
        printHelp()
        sys.exit()
    elif opt in ('-o', '--order'):
        order = arg
    elif opt in ('-m', '--mfile'):
        corpusOutputFile = arg
    elif opt in ('-d', '--dfile'):
        dictionaryOutputFile = arg
    elif opt in ('-e', '--efile'):
        evaluationOutputFile = arg
    elif opt in ('-E', '--evaluateonly'):
        evaluateonly = True
    elif opt in ('-V', '--vocabulary'):
        vocabularyInputFile = arg
    elif opt in ('-l', '--files'):
        with open(arg, 'r') as f:
            for line in f:
                generateCountsFor.append(line.rstrip())
    elif opt in ('-c', '--count'):
        with open(arg, 'r') as f:
            for line in f:
                inputFiles.append(line.rstrip())
    elif opt in ('-C', '--countdir'):
        countOutputDirectory = arg
    elif opt in ('-v', '--verbose'):
        verbosity = int(arg)

### Generate count files ########################

if isinstance(int(order), (int, long)):
    print "isint"
else:
    print "notint"

if len(generateCountsFor) and order.isdigit():
    condPrint(2, " > Generating %d count files" % len(generateCountsFor))
    countStart = time.time()
    for line in generateCountsFor:
        createCountsFor(line)
    condPrint(5, " < Done generating count files in %f" % (time.time() - countStart))

### Output parameters ###########################

condPrint(2, "-- Order: %s" % order)
condPrint(2, "-- Dictionary file: %s" % dictionaryOutputFile)
condPrint(2, "-- Corpus file: %s" % corpusOutputFile)
condPrint(2, "-- Vocabulary file: %s" % vocabularyInputFile)
condPrint(2, "-- Counts directory: %s" % countOutputDirectory)
condPrint(2, "-- Number of input files: %d" % len(inputFiles))
condPrint(2, "-- Number of input files to process: %d" % len(generateCountsFor))
condPrint(2, "-- Verbosity level: %d" % verbosity)

### Evaluation output ###########################

if evaluationOutputFile:
    condPrint(2, ">  Processing evaluation output file")
    if order == 'n':
        condPrint(2, " - Creating sentence histories")
        createSentenceHistories()
    else:
        condPrint(2, " - Creating evaluation file")
        createEvaluationFile()
    condPrint(2, "<  Done processing evaluation output file")

    if evaluateonly:
        condPrint(2, "-- Evaluate only mode enabled. Done")
        sys.exit()
else:
    condPrint(2, "-- Skipping evaluation output")

### Vocabulary ##################################

condPrint(2, "-- Processing vocabulary")
vocabStart = time.time()
if vocabularyInputFile:
    condPrint(2, ">  Reading vocabulary")
    readVocabularyAsDictionary()
    nrTrainInstances = countTrainInstances()
else:
    condPrint(2, ">  Creating dictionary with open vocabulary")
    dictStart = time.time()
    nrTrainInstances = createOpenDictionary()
condPrint(4, " - Processed dictionary/vocabulary in %f seconds" % (time.time() - vocabStart))
condPrint(2, "<  Processed dictionary/vocabulary with %d words" % len(dictionary))

###

dictionaryCounts = [0] * len(dictionary)

### Corpus output ###############################

if corpusOutputFile:
    condPrint(2, ">  Writing corpus to %s" % (corpusOutputFile)) 
    cofStart = time.time()
    cof = open(corpusOutputFile, 'w')
    
    cof.write("%%MatrixMarket matrix coordinate real general\n")
    cof.write("%============================================\n")
    cof.write("% Generated for the files:                   \n")
    for inputFile in inputFiles:
        cof.write("% " + inputFile + "\n");
    cof.write("%============================================\n")
    cof.write("%d %d %d\n" % (len(inputFiles), len(dictionary), nrTrainInstances))

    fileNumber = 1
    for inputFile in inputFiles:
        with open(inputFile, 'r') as f:
            for line in f:
                (word, value) = line.rstrip().split('\t')
                if word in dictionary:
                    dictionaryCounts[dictionary[word]-1] += int(value)
                    cof.write("%d %d %d\n" % (fileNumber, dictionary[word], int(value)))
            fileNumber += 1
    cof.close()
    condPrint(4, " - Done writing corpus in %f seconds" % (time.time() - cofStart))
    condPrint(5, "<  Done writing corpus")

### Dictionary output ##########################

if dictionaryOutputFile:
    condPrint(2, ">  Writing dictionary to %s" % (dictionaryOutputFile))
    dofStart = time.time()
    dof = open(dictionaryOutputFile, 'w')
    for key in dictionary:
        dof.write("%d\t%s\t%d\n" % (dictionary[key], key, dictionaryCounts[dictionary[key]-1]))
    dof.close()
    condPrint(4, " - Wrote dictionary in %f seconds" % (time.time() - dofStart))
    condPrint(5, "<  Done writing dictionary")
