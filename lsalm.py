#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import gensim
import numpy
import math
import sys
import getopt
import time
import pickle
from time import gmtime, strftime
from colorama import init, Fore, Back, Style
from threading import Lock, Thread
from multiprocessing import Process, Queue, Manager

import pprint

class PrintLevel:
    (NORMAL, GENERAL, STEPS, TIME, SPECIFIC, EVERYTHING) = range(0, 6)

class InvalidValueError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class LsaLM:

    minCosCache = {} # contains the min cos cache of a context
    corpusCounts = {} # contains the corpus count of a word id
    LSAconfCacheNoms = {} # contains the lsa confidence value of a word id
    contextCentroids = {} # contains the centroid vector of a context
    PLEWordIdContext = {} # contains the PL estimate value for a word id and a context
    sumPLEPerContext = {} # contains the PL estimate sum for a context
    sumLSAConfPLEPerContext = {} # contains the sum(PL(j)*LsaConf(j)) for a context
    srilmContext = {} # contains 
    sumLSAConfPBEPerContext = {}
    srilmProbs = {}

    def printVariableLengths(self):
        print "%s %d [%d], %s %d, %s %d, %s %d, %s %d, %s %d [%d], %s %d, %s %d, %s %d" % ("mcc", len(self.minCosCache), sys.getsizeof(pickle.dumps(self.minCosCache)),  "cc", len(self.corpusCounts), "lccn", len(self.LSAconfCacheNoms), "cxc", len(self.contextCentroids), "pwid", len(self.PLEWordIdContext), "spc", len(self.sumPLEPerContext), sys.getsizeof(pickle.dumps(self.sumPLEPerContext)), "slcpc", len(self.sumLSAConfPLEPerContext), "sc", len(self.srilmContext), "sp", len(self.srilmProbs))

    sumPLEPerContextLock = Lock()
    PLEWordIdContextLock = Lock()
    minCosCacheLock = Lock()
    
    verbosity = int(0)
    programIdentifier = ''
    threads = 24
    
    distributed = False
    mmfile = 'markovmarket'
    dictfile = 'dictionary'
    normfile = ''
    gamma = float(7.0)
    dimensions = 150
    trainFile = ''

    srilmProbsDirectory = ''
    contextIndex = ''

    parallelPLEsFile = ''
    plainPLEsFile = ''
    parallelLSAConfPLEsFile = ''
    plainLSAConfPLEsFile = ''

    saveLSIFile = ''
    readLSIFile = ''
    outputFile = ''
    readContextPLEsFile = ''
    writeContextPLEsFile = ''
    readContextLSAConfPLEsFile = ''
    writeContextLSAConfPLEsFile = ''
    readContextLSAConfPBEsFile = ''
    writeContextLSAConfPBEsFile = ''
    readMinCosFile = ''
    writeMinCosFile = ''
    readContextsFile = ''
    writeContextsFile = ''
    readWordCountFile = ''
    writeWordCountFile = ''
    readLSAConfFile = ''
    writeLSAConfFile = ''
    readNormalisationFile = ''
    writeNormalisationFile = ''

    taskParts = 100
    evaluatePart = int(-1)
  
 
    def printHelp(self):
        self.condPrint(PrintLevel.EVERYTHING, "The LSA Language Model comprises 7 steps. Several steps may take a lot of time,", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "therefore it offers functionality to save intermediate results.", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-h, --help                 print this text and exit", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-i, --id s                 give the program an id, useful for output of parallel processes", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-v, --verbosity n          set verbosity level (default=0)", addPrefix=False)    
        self.condPrint(PrintLevel.NORMAL, "                           (0=normal, 1=general, 2=steps, 3=time, 4=specific, 5=everything)", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-I, --threads n            the number of threads used in parallel computations (default=24)", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "-- Dictionary and Corpus", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "First, the dictionary and corpus are loaded. Even if the LSA Space created from", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "these files is saved, dictionary and corpus files are mandatory. See createIndex.py", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "on how to create these files", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-m, --mfile f              read corpus file from f (in matrix market format)", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-d, --dfile f              read dictionary from f", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "-- LSA Space", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "Then a LSA Space is built. We use gensim, so make sure it is installed. If you want", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "to run a distributed version, install it according to http://radimrehurek.com/gensim/dist_lsi.html", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "If -x is provided, it reads the model from file, otherwise a new model is computed.", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-D, --distributed          use the distributed version of gensim", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-k, --dimensions n         the number of dimensions after SVD (default=150)", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-w, --write f              write LSA model to file f", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-r, --read f               read LSA from file f", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "-- Word Counts", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "To avoid counting the words over and over again, we cache the word counts. This", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "cache can also be saved to file with -X. If -x is provided, the word counts are.", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "retrieved from the file")
        self.condPrint(PrintLevel.NORMAL, "-c, --readcount f          read word count file from f", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-C, --writecount f         write word count to file f", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "-- LSA Confidence Values", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "The LSA Confidence Values can be precomputed for a corpus", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-l, --readlsaconf f        read lsa confidence values from f", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-L, --writelsaconf f       write lsa condifence values to f", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "-- Read Contexts", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "In many stages we need a mapping between contexts and their centroids in the LSA", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "Space. Therefore we read all the contexts and centroids beforehand.", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-x, --readcontexts f       read contexts from f", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-X, --writecontexts f      write contexts to f", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "-- LSA Probability Estimates (PLEs)", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "The LSA LM uses a technique to increase the dynamic range, with parameter gamma.", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "To normalise the probabilities based on these estimates, we have to sum over the", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "context probabilities: PLEs. If you want to computes these in parallel, use -Q. This", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "will output tab-separated (plain) output to a file, which can be compined with other", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "-Q's, with a simple concatenation. This can then be read with -q. Generating these", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "PLEs generally takes a long time. However, if you want to run it with 1 process, use", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "-P and -p. These are binary formats. You can read with -q and save with -P to a", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "binary file. If you use -Q, you can choose which part you want to do in the process.", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "Choose part n of 100 with -e n, (or of 1000 if -T -e n).", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-g, --gamma n              gamma parameter for dynamic range scaling (default=7.0)", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-Q, --parallelple f        write intermediate step to f", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-q, --plainples f          read plain pl estimates (e.g. multiple -Q)", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-p, --readples f           read context pl estimates from f", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-P, --writeples f          write context pl estimates to file f", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-T, --thousand             divide task in 1000 pieces, rather than the default 100", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-e, --evaluatepart n       evaluate subset n (of 100, unless -T), all=-1 (default=-1)", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "-- Mincos", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "Since the mincos (and related) values are not dependent of the words, we can precompute them.", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-z, --readmincos f         read the mincos values from f", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-Z, --writemincos f        write the mincos values to f", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "-- LSA Probability multiplied with LSA Confidence value", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "Similar to the previous block, but now the probabilities are not longer", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "estimates, and we multiply the values with their corresponding lsa confidence", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "-e can still be used", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-B, --parallellcple f      write intermediate lc pl step to f", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-b, --plainlcples f        read plain lc pl estimates (e.g. multiple -Q)", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-j, --readlcples f         read context lc pl estimates from f", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-J, --writelcples f        write context lc pl estimates to file f", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "-- Applying the rules", addPrefix=False)
        self.condPrint(PrintLevel.EVERYTHING, "The last phase is about applying the rules to new data", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-S, --srilm f              contains the srilm n-gram logprobs (tab separated)")
        self.condPrint(PrintLevel.NORMAL, "-t, --test f               read lines to apply trained model on", addPrefix=False)
        self.condPrint(PrintLevel.NORMAL, "-s, --save f               save output (probability and lsa confidence) to file f", addPrefix=False)

    def cos(self,wordId,context):
        w = self.lsi.projection.u[wordId]
        C = self.contextCentroids[context]

        nom = numpy.dot(w,C)
        det1 = numpy.linalg.norm(w)
        det2 = numpy.linalg.norm(C)
        if math.isnan(nom) or math.isnan(det1) or math.isnan(det2) or det1 == 0 or det2 == 0 or nom == 0:
            return 0
        if det1*det2 <= 0:
            raise InvalidValueError(Fore.RED + "The normalisation of the cos function shouldn't be <= 0!\nValue: %f\nWord: %s\nContext: %s" % (det1*det2, self.id2word[wordId], context))
        # FORMULA (2)
        val = nom/(det1*det2)
        return 0 if math.isnan(val) else val
    
    def minCos(self,context):
        minVal = 0
        minId = 0
        cosSum = 0
        for wId in self.id2word:
            val = self.cos(wId, context)
            cosSum += val
            # FORMULA (3)
            if(val < minVal):
                minVal = val
                minId = wId
            
        plDen = cosSum - len(self.id2word) * minVal
        return (minId, minVal, cosSum, plDen)
    
    def getCachedminCos(self,context,mcc=None):
        if not mcc:
            mcc = self.minCosCache
        result = 0
        if context in mcc:
            result = mcc[context]
        else:
            mincos = self.minCos(context)
            mcc[context] = mincos
            pprint.pprint(mcc)
            result = mincos
        return result

    def getCachedPLE(self,wordId,context,cache=True):
        comboString = ("%d %s" % (wordId, context))
        self.PLEWordIdContextLock.acquire()
        if comboString in self.PLEWordIdContext:
            result = self.PLEWordIdContext[comboString]
            self.PLEWordIdContextLock.release()
            return result
        else:
            self.PLEWordIdContextLock.release()
            (minId, minVal, cosSum, plDen) = self.getCachedminCos(context)
            wordCos = self.cos(wordId, context)
            if wordCos < minVal:
                raise InvalidValueError(Fore.RED + "Word cosine in a context cannot be smaller than the smallest value in the context!\ncosine value (and smallest): %f (%f)\nWord: %s\nContext: %s" % (wordCos, minVal, self.id2word[wordId], context))
            # FORMULA (4)
            PLest = (wordCos - minVal) / plDen
            if PLest < 0:
                raise InvalidValueError(Fore.RED + "PLest < 0\nWord: %s\nContext: %s\nWordId lowest cosine with value:%d (%s) %f\nCossum:%f\nplDen: %f" % (self.id2word[wordId], context, minId, self.id2word[minId], minVal, cosSum, plDen))

            if cache:
                self.PLEWordIdContextLock.acquire()
                self.PLEWordIdContext[comboString] = PLest
                self.PLEWordIdContextLock.release()
            return PLest

    def parallelLCPBE(sef, info, queue, d, l):
        while True:
            info, context = queue.get()
            if not info and not context:
                return

            if info and not context:
                self.condPrint(PrintLevel.GENERAL, "Processing %dth element in process %d" % (info, pId))
            else:
                for wId in self.id2word:
                    lc = self.getPrecachedLSA

    # Computes the part below the divider in formula (7)
    def parallelLCPLE(self, info, queue, d, l):
        while True:
            info, context = queue.get()
            if not info and not context:
                return

            if info and not context:
                self.condPrint(PrintLevel.GENERAL, "Processing %d th element in process %d" % (info, pId))
            else:
                for wId in self.id2word:
                    lc = self.getPrecachedLSAConf(wId)
                    l.acquire()
                    d[context] = d.get(context, 0) + pow(self.PL(wId, context), lc) + pow(self.getSRILMProb(wId, context,lc), 1-lc)
                    l.release()
                    #self.sumPLEPerContext[context] = self.sumPLEPerContext.get(context, 0) + pow(PLEst, self.gamma)
#self.sumLSAConfPLEPerContext[context] = self.sumLSAConfPLEPerContext.get(context, 0) + pow(self.PL(wId, context), self.getPrecachedLSAConf(wId)) + pow(self.getSRILMProbs(context), 1-self.getPrecachedLSAConf(wId))


        comboString = ("%d %s" % (wordId, context))
            (minId, minVal, cosSum, plDen) = self.getCachedminCos(context)
            wordCos = self.cos(wordId, context)
            # FORMULA (4)
            PLest = (wordCos - minVal) / plDen




    def parallelPLE(self, pId, queue, d, l):
        while True:
            wId, context = queue.get()
            if not wId and not context:
                return

            if wId and not context:
                if wId < 0:
                    self.condPrint(PrintLevel.GENERAL, "Processing %dth context in process %d" % (wId, pId))
                    #self.printVariableLengths()
                else:
                    self.condPrint(PrintLevel.GENERAL, "Processing %dth element in process %d" % (wId, pId))
            else:
                PLEst = self.getCachedPLE(wId, context,cache=False)
                l.acquire()
                d[context] = d.get(context, 0) + pow(PLEst, self.gamma)
                l.release()
    
    def parallelMinCos(self, pId, queue, d):
        while True:
            cId, context = queue.get()
            if not cId and not context:
                return

            result = 0
            if context in d:
                result = d[context]
            else:
                mincos = self.minCos(context)
                d[context] = mincos
                result = mincos

    def PL(self,wordId,context):
        # FORMULA (5)
        return pow(self.getCachedPLE(wordId, context), self.gamma) / self.sumPLEPerContext[context]
    
    def getWordCountById(self,wordId):
        return self.corpusCounts.get(wordId, 0)

    def buildLSAConfCache(self):
        for doc in self.mm:
            for wId,val in doc:
                gwc = self.getWordCountById(wId)
                if gwc:
                    Pij = val/gwc
                    self.LSAconfCacheNoms[wId] = self.LSAconfCacheNoms.get(wId, 0) + Pij * math.log(Pij)
        for keyId, key in enumerate(self.LSAconfCacheNoms):
            self.LSAconfCacheNoms[key] = 1+ self.LSAconfCacheNoms[key]/math.log(self.mm.num_docs)
    
    def getPrecachedLSAConf(self,wId):
        # LSAconfCacheNoms starts with 0
        # The rest starts with 1
        return self.LSAconfCacheNoms.get(wId-1, 0)

    def getSRILMProbs(self, context):
        contextId = contextsToId[context]
        with open("%s/context.%d" % (self.srilmProbsDirectory, contextId), 'r') as f:
            contextSum = 0
            for line in f:
                context, logprob = line.rstrip().split('\t')
                prob = math.exp(logprob)
                contextSum += prob
            srilmContext[context] = contextSum

    def getSRILMProb(self, wId, context, lsaconf=1.0):
        word = self.id2word[wId]
        contextId = contextsToId[context]
        result = 0
#        if context not in srilmContext:
        with open("%s/context.%d" % (self.srilmProbsDirectory, contextId), 'r') as f:
            contextSum = 0
            for line in f:
                localContext, logprob = line.rstrip().split('\t')
                if context == localContext:
                    result = math.exp(logprob)
                prob = math.exp(logprob)
                contextSum += pow(prob, 1-lsaconf)
            srilmContext[context] = contextSum
        return result

    def getWordCounts(self):
        for doc in self.mm:
            for wId, val in doc:
                self.corpusCounts[wId] = self.corpusCounts.get(wId, 0) + val
   
    def condPrint(self,level,text,name='',addPrefix=True):
        if level <= self.verbosity:
            prefix = ""
            if addPrefix:
                timeStr = strftime("%Y%m%d %H:%M:%S", gmtime())
                if self.programIdentifier:
                    prefix = '%s %10s ' % (timeStr, self.programIdentifier)
                elif name:
                    prefix = '%s %10s ' % (timeStr, name)
                else:
                    prefix = '%s ' % (timeStr)

            print prefix + text

    def __init__(self, cmdArgs):
        try:
            opts, args = getopt.getopt(cmdArgs, 'hi:I:Dm:x:X:d:g:k:t:w:r:s:v:c:C:l:L:n:N:Te:p:P:Q:q:j:J:B:b:S:I:z:Z:r:R:', ['help', 'id=', 'threads=', 'distributed', 'mfile=', 'readcontexts=', 'writecontexts', 'dfile=', 'gamma=', 'dimensions=','test=', 'write=', 'read=', 'save=', 'verbosity=', 'readcount=', 'writecount=', 'readlsaconf=', 'writelsaconf=', 'thousand', 'evaluatepart=', 'readples=', 'writeples=', 'parallelple=', 'plainples=', 'readlcples=', 'writelcples=', 'parallellcple=', 'plainlcples=', 'srilm=', 'contextindex=', 'readmincos=', 'writemincos=', 'readlcpbes=', 'writelcpbes=' ])
        except getopt.GetoptError:
            self.printHelp()
            sys.exit(2)
        
        for (opt, arg) in opts:
            if opt in ('-h', '--help'):
                self.verbosity = (5)
                self.printHelp()
                sys.exit()
            if opt in ('-i', '--id'):
                self.programIdentifier = arg
            if opt in ('-I', '--threads'):
                self.threads = int(arg)
            elif opt in('-D', '--distributed'):
                self.distributed = True
            elif opt in ('-m', '--mfile'):
                self.mmfile = arg
            elif opt in ('-x', '--readcontexts'):
                self.readContextsFile = arg
            elif opt in ('-X', '--writecontexts'):
                self.writeContextsFile = arg
            elif opt in ('-d', '--dfile'):
                self.dictfile = arg
            elif opt in ('-g', '--gamma'):
                self.gamma = float(arg)
            elif opt in ('-k', '--dimensions'):
                self.dimensions = arg
            elif opt in ('-t', '--train'):
                self.trainFile = arg
            elif opt in ('-w', '--write'):
                self.saveLSIFile = arg
            elif opt in ('-r', '--read'):
                self.readLSIFile = arg
            elif opt in ('-s', '--save'):
                self.outputFile = arg
            elif opt in ('-v', '--verbosity'):
                self.verbosity = int(arg)
            elif opt in ('-c', '--readcount'):
                self.readWordCountFile = arg
            elif opt in ('-C', '--writecount'):
                self.writeWordCountFile = arg
            elif opt in ('-l', '--readlsaconf'):
                self.readLSAConfFile = arg
            elif opt in ('-e', '--evaluatepart'):
                self.evaluatePart = int(arg)
            elif opt in ('-T', '--thousand'):
                self.taskParts = 1000
            elif opt in ('-L', '--writelsaconf'):
                self.writeLSAConfFile = arg
            elif opt in ('-p', '--readples'):
                self.readContextPLEsFile = arg
            elif opt in ('-P', '--writeples'):
                self.writeContextPLEsFile = arg
            elif opt in ('-Q', '--parallelple'):
                self.parallelPLEsFile = arg
            elif opt in ('-q', '--plainples'):
                self.plainPLEsFile = arg
            elif opt in ('-j', '--readlcples'):
                self.readContextLSAConfPLEsFile = arg
            elif opt in ('-J', '--writelcples'):
                self.writeContextLSAConfPLEsFile = arg
            elif opt in ('-B', '--parallellcple'):
                self.parallelLSAConfPLEsFile = arg
            elif opt in ('-b', '--plainlcples'):
                self.plainLSAConfPLEsFile = arg
            elif opt in ('-S', '--srilm'):
                self.srilmProbsDirectory = arg
            elif opt in ('-I', '--contextindex'):
                self.contextIndex = arg
            elif opt in ('-z', '--readmincos'):
                self.readMinCosFile = arg
            elif opt in ('-Z', '--writemincos'):
                self.writeMinCosFile = arg
            elif opt in ('-r', '--readpcpbes'):
                self.readContextLSAConfPBEsFile = arg
            elif opt in ('-R', '--writepcpbes'):
                self.writeContextLSAConfPBEsFile = arg
       
        init(autoreset=True)

        self.condPrint(PrintLevel.GENERAL, "Program identifier: %s" % self.programIdentifier)
        self.condPrint(PrintLevel.GENERAL, "Number of threads: %d" % self.threads)
        self.condPrint(PrintLevel.GENERAL, "Corpus file: %s" % self.mmfile)
        self.condPrint(PrintLevel.GENERAL, "Distributed: %s" % ("Yes" if self.distributed else "No"))
        self.condPrint(PrintLevel.GENERAL, "Dictionary file: %s" % self.dictfile)
        self.condPrint(PrintLevel.GENERAL, "Read contexts from: %s" % self.readContextsFile)
        self.condPrint(PrintLevel.GENERAL, "Write contexts to: %s" % self.writeContextsFile)
        self.condPrint(PrintLevel.GENERAL, "Read word counts from: %s" % self.readWordCountFile)
        self.condPrint(PrintLevel.GENERAL, "Write word counts to: %s" % self.writeWordCountFile)
        self.condPrint(PrintLevel.GENERAL, "Read LSA confidence values from: %s" % self.readLSAConfFile)
        self.condPrint(PrintLevel.GENERAL, "Write LSA confidence values to: %s" % self.writeLSAConfFile)
        self.condPrint(PrintLevel.GENERAL, "Gamma: %f" % self.gamma)
        self.condPrint(PrintLevel.GENERAL, "Dimensions: %s" % self.dimensions)
        self.condPrint(PrintLevel.GENERAL, "Evaluate on: %s" % self.trainFile)
        self.condPrint(PrintLevel.GENERAL, "SRILM n-gram probabilities in: %s" % self.srilmProbsDirectory)
        self.condPrint(PrintLevel.GENERAL, "Save LSA in: %s" % self.saveLSIFile)
        self.condPrint(PrintLevel.GENERAL, "Read LSA from: %s" % self.readLSIFile)
        self.condPrint(PrintLevel.GENERAL, "Read Context PL estimates from: %s" % self.readContextPLEsFile)
        self.condPrint(PrintLevel.GENERAL, "Write Context PL estimates to: %s" % self.writeContextPLEsFile)
        self.condPrint(PrintLevel.GENERAL, "Write tab-separated PLEs to f")
        self.condPrint(PrintLevel.GENERAL, "Read from tab-separated PLEs from f")
        self.condPrint(PrintLevel.GENERAL, "Write output to: %s" % self.outputFile)
        self.condPrint(PrintLevel.GENERAL, "Write mincos to: %s" % self.writeMinCosFile)
        self.condPrint(PrintLevel.GENERAL, "Read mincos from: %s" % self.readMinCosFile)
        self.condPrint(PrintLevel.GENERAL, "Read context-index SRILM files from: %s" % self.srilmProbsDirectory)
        self.condPrint(PrintLevel.GENERAL, "Evaluate only part %s of %s" % (self.evaluatePart, self.taskParts))
        self.condPrint(PrintLevel.GENERAL, "Verbosity level: %s" % self.verbosity)

    def buildSpace(self):
        if self.outputFile:
            self.of = open(self.outputFile, 'w')
        
        ### Dictionary and Corpus #######################
        
        self.condPrint(PrintLevel.GENERAL, "-- Processing dictionary and corpus")

        self.condPrint(PrintLevel.STEPS, ">  Reading dictionary")
        readDictStart = time.time()
        self.id2word = gensim.corpora.Dictionary.load_from_text(self.dictfile)
        #self.condPrint(PrintLevel.TIME, self.id2word)
        self.condPrint(PrintLevel.TIME, " - Took %f seconds" % (time.time() - readDictStart))
        self.condPrint(PrintLevel.STEPS, "<  Done reading dictionary")

        
        self.condPrint(PrintLevel.STEPS, ">  Reading corpus")
        readCorpusStart = time.time()
        self.mm = gensim.corpora.MmCorpus(self.mmfile)
        #self.condPrint(PrintLevel.TIME, self.mm)
        self.condPrint(PrintLevel.TIME, " - Took %f seconds" % (time.time() - readCorpusStart))
        self.condPrint(PrintLevel.STEPS, "<  Done reading corpus")
        
        ### LSA Space ###################################
        
        self.condPrint(PrintLevel.GENERAL, "-- Processing LSA Space")

        self.lsi = None
        lsiStart = time.time()
        if not self.readLSIFile:
            self.condPrint(PrintLevel.STEPS, ">  Generating LSA model")
            self.lsi = gensim.models.lsimodel.LsiModel(corpus=self.mm, id2word=self.id2word, num_topics=self.dimensions, distributed=self.distributed)
        else:
            self.condPrint(PrintLevel.STEPS, ">  Reading LSA model")
            self.lsi = gensim.models.lsimodel.LsiModel.load(self.readLSIFile)
        self.condPrint(PrintLevel.TIME, " - Took %f seconds" % (time.time() - lsiStart))
        self.condPrint(PrintLevel.STEPS, "<  Done processing the LSA model")
        
        if self.saveLSIFile and not self.readLSIFile:
            self.condPrint(PrintLevel.STEPS, ">  Writing LSA model to file")
            lsaWStart = time.time()
            self.lsi.save(self.saveLSIFile)
            self.condPrint(PrintLevel.TIME, " - Took %f seconds"  % (time.time() - lsaWStart))
            self.condPrint(PrintLevel.STEPS, "<  Done writing LSA model to file: %s" % self.saveLSIFile)
        
        ### Word Counts #################################
        
        self.condPrint(PrintLevel.GENERAL, "-- Processing Word Counts")

        countStart = time.time()
        if not self.readWordCountFile:
            self.condPrint(PrintLevel.STEPS, ">  Counting words")
            self.getWordCounts()
        else:
            self.condPrint(PrintLevel.STEPS, ">  Reading word counts")
            wcFile = open(self.readWordCountFile, 'rb')
            corpusCounts = pickle.load(wcFile)
            wcFile.close()
        self.condPrint(PrintLevel.TIME, " - Took %f seconds" % (time.time() - countStart))
        self.condPrint(PrintLevel.STEPS, "<  Done counting words")
        
        if self.writeWordCountFile and not self.readWordCountFile:
            self.condPrint(PrintLevel.STEPS, ">  Writing word counts to file")
            wcWriteStart = time.time()
            wcFile = open(self.writeWordCountFile, 'wb')
            pickle.dump(self.corpusCounts, wcFile)
            wcFile.close()
            self.condPrint(PrintLevel.TIME, " - Took %f seconds" % (time.time() - wcWriteStart))
            self.condPrint(PrintLevel.STEPS, "<  Done writing to file")
        
        ### LSA Confidence ##############################
        
        self.condPrint(PrintLevel.GENERAL, "-- Processing LSA Confidence values")

        lsaconfStart = time.time()
        if not self.readLSAConfFile:
            self.condPrint(PrintLevel.STEPS, ">  Computing LSA confs")
            self.buildLSAConfCache()
        else:
            self.condPrint(PrintLevel.STEPS, ">  Reading LSA confs")
            lcFile = open(self.readLSAConfFile, 'rb')
            self.LSAconfCacheNoms = pickle.load(lcFile)
            wcFile.close()
        self.condPrint(PrintLevel.TIME, " - Processing LSA confs took %f seconds" % (time.time() - lsaconfStart))
        self.condPrint(PrintLevel.STEPS, "<  Done reading LSA confs")
        
        if self.writeLSAConfFile and not self.readLSAConfFile:
            self.condPrint(PrintLevel.STEPS, ">  Writing LSA confidence counts to file")
            lcWriteStart = time.time()
            lcFile = open(self.writeLSAConfFile, 'wb')
            pickle.dump(self.LSAconfCacheNoms, lcFile)
            lcFile.close()
            self.condPrint(PrintLevel.TIME, " - Writing LSA confs took %f seconds" % (time.time() - lcWriteStart))
            self.condPrint(PrintLevel.STEPS, "<  Done writing to file")

        ### Read Contexts ###############################

        self.condPrint(PrintLevel.GENERAL, "-- Processing contexts")

        readCStart = time.time()
        if self.readContextsFile:
            self.condPrint(PrintLevel.STEPS, ">  Reading contexts from file")
            rcFile = open(self.readContextsFile, 'rb')
            self.contextCentroids = pickle.load(rcFile)
            rcFile.close()
        else:
            if self.trainFile:
                self.condPrint(PrintLevel.STEPS, ">  Generating contexts")
                with open(self.trainFile, 'r') as f:
                    for line in f:
                        self.createCentroid(line.rstrip())
            else:
                self.condPrint(PrintLevel.STEPS, ">  No train file is given, therefore no contexts are generated!")
        self.condPrint(PrintLevel.TIME, " - Generating contexts took %f seconds" % (time.time() - readCStart))
        self.condPrint(PrintLevel.STEPS, "<  Done reading contexts")

        if self.writeContextsFile and not self.readContextsFile:
            self.condPrint(PrintLevel.STEPS, ">  Writing contexts to file")
            cWriteStart = time.time()
            rcFile = open(self.writeContextsFile, 'wb')
            pickle.dump(self.contextCentroids, rcFile)
            rcFile.close()
            self.condPrint(PrintLevel.TIME, " - Writing contexts took %f seconds" % (time.time() - cWriteStart))
            self.condPrint(PrintLevel.STEPS, "<  Done writing contexts to file")

        ### MinCos ######################################

        self.condPrint(PrintLevel.STEPS, "-- Precomputing the mincos values")
       
        mcStart = time.time()
        if self.readMinCosFile:
            self.condPrint(PrintLevel.STEPS, ">  Reading mincos from file")
            mcFile = open(self.readMinCosFile, 'rb')
            self.minCosCache = pickle.load(mcFile)
            #with open(self.readMinCosFile, 'r') as f:
            #    for line in f:
            #        l, r, a, b, c, d = line.rstrip().split('\t')
            #        context = "%s\t%s" % (l, r)
            #        self.minCosCache[context] = (a, b, c, d)
            mcFile.close()
        else:
            mcManager = Manager()
            mcD = mcManager.dict()

            mcQueue = Queue()
            processes = []

            for i in range(self.threads):
                process = Process(target=self.parallelMinCos, args=(i, mcQueue,mcD,))
                processes.append(process)
                process.start()

            for contextId, context in enumerate(self.contextCentroids):
                mcQueue.put((contextId, context))
                #if contextId > 30:
                #    break

            for _ in range(self.threads):
                mcQueue.put((None,None))

            for process in processes:
                process.join()

            print "%d elements in mcD" % len(mcD)
            #self.minCosCache = mcD
            self.minCosCache = {}
            for k in mcD.keys():
                self.minCosCache[k] = mcD[k]
            print "%d elements in mcc" % len(self.minCosCache)
        self.condPrint(PrintLevel.TIME, " - Reading mincos took %f seconds" % (time.time() - mcStart))
        self.condPrint(PrintLevel.STEPS, "<  Done reading the mincos values")

        if self.writeMinCosFile and not self.readMinCosFile:
            self.condPrint(PrintLevel.STEPS, ">  Writing mincos to file")
            mcStart = time.time()
            mcFile = open(self.writeMinCosFile, 'wb')
            pickle.dump(self.minCosCache, mcFile)
            #for k in self.minCosCache.keys():
            #    (a, b, c, d) = self.minCosCache.get(k, (0,0,0,0))
            #    mcFile.write("%s\t%.16f\t%.16f\t%.16f\t%.16f\n" % (k, a, b, c, d))
            mcFile.close()
            self.condPrint(PrintLevel.TIME, " - Writing mincos took %f seconds" % (time.time() - mcStart))
            self.condPrint(PrintLevel.STEPS, "<  Done writing mincos to file")

        ### PLEs ########################################

        self.condPrint(PrintLevel.GENERAL, "-- Processing the PLEs")

        cpStart = time.time()
        if self.readContextPLEsFile:
            self.condPrint(PrintLevel.STEPS, ">  Reading Context PLs from file")
            cpFile = open(self.readContextPLEsFile, 'rb')
            self.sumPLEPerContext = pickle.load(cpFile)
            cpFile.close()
        elif self.plainPLEsFile:
            self.condPrint(PrintLevel.STEPS, ">  Reading tab-separated Context PLs from file")
            with open(self.plainPLEsFile, 'r') as f:
                for line in f:
                    lContext, rContext, value = line.rstrip().split('\t')
                    context = lContext + '\t' + rContext
                    self.sumPLEPerContext[context] = float(value)
        else:
            self.condPrint(PrintLevel.STEPS, ">  Computing Context PLs")

            self.condPrint(PrintLevel.NORMAL, " >>> Starting parallel processing of PLEs")


            pleManager = Manager()
            pleD = pleManager.dict()
            pleL = pleManager.Lock()
            pleQueue = Queue()
            processes = []

            for i in range(self.threads):
                process = Process(target=self.parallelPLE, args=(i,pleQueue,pleD,pleL))
                processes.append(process)
                process.start()

            placeInQueue = 0
            for contextId, context in enumerate(self.contextCentroids):
                pleQueue.put((-contextId, None))
                for wId in self.id2word:
                    #self.condPrint(PrintLevel.GENERAL, "<-x-> %d" % placeInQueue)
                    if not placeInQueue % 10000:
                       pleQueue.put((placeInQueue, None))
                    pleQueue.put((wId, context))
                    placeInQueue += 1

            self.condPrint(PrintLevel.GENERAL, "Putting an end to this shit!")
            for _ in range(self.threads):
                pleQueue.put((None, None))
            
            for process in processes:
                process.join()

            #self.sumPLEPerContext = pleD
            sumPLEPerContext = {}
            for k in pleD.keys():
                sumPLEPerContext[k] = pleD[k]

            self.condPrint(PrintLevel.NORMAL, " <<< Done with parallel processing of PLEs")

        self.condPrint(PrintLevel.TIME, " - Reading/computing Context PLs took %f seconds" % (time.time() - cpStart))
        self.condPrint(PrintLevel.STEPS, "<  Done reading/computing Context PLs")

        if self.writeContextPLEsFile and not self.readContextPLEsFile:
            self.condPrint(PrintLevel.STEPS, ">  Writing Context PLs to file")
            cWriteStart = time.time()
            cpFile = open(self.writeContextPLEsFile, 'wb')
            pickle.dump(self.sumPLEPerContext, cpFile)
            cpFile.close()
            self.condPrint(PrintLevel.TIME, " - Writing Context PLs took %f seconds" % (time.time() - cWriteStart))
            self.condPrint(PrintLevel.STEPS, "<  Done writing Context PLs to file")


        print "-------------- done ------------------"
        sys.exit()

        ### LC PL STUFF #################################

        self.condPrint(PrintLevel.GENERAL, "-- Processing the LSA conf PLEs")

        cpStart = time.time()
        if self.readContextLSAConfPLEsFile:
            self.condPrint(PrintLevel.STEPS, ">  Reading Context LC PLs from file")
            cpFile = open(self.readContextLSAConfPLEsFile, 'rb')
            self.sumLSAConfPLEPerContext = pickle.load(cpFile)
            cpFile.close()
        elif self.plainLSAConfPLEsFile:
            self.condPrint(PrintLevel.STEPS, ">  Reading tab-separated Context LC PLs from file")
            with open(self.plainLSAConfPLEsFile, 'r') as f:
                for line in f:
                    lContext, rContext, value = line.rstrip().split('\t')
                    context = lContext + '\t' + rContext
                    self.sumLSAConfPLEPerContext[context] = float(value)
        else:
            self.condPrint(PrintLevel.STEPS, ">  Computing Context LC PLs")

            self.condPrint(PrintLevel.NORMAL, " >>> Starting parallel processing of PLEs")

            lcpleManager = Manager()
            lcpleD = lcpleManager.dict()
            lcpleL = lcpleManager.Lock()
            lcpleQueue = Queue()
            processes = []

            for i in range(self.threads):
                process = Process(target=self.parallelLCPLE, args=(i,lcpleQueue,lcpleD,lcpleL,))
                processes.append(process)
                process.start()

            placeInQueue = 0
            for contextId, context in enumerate(self.contextCentroids):
                for wId in self.id2word:
                    #self.condPrint(PrintLevel.GENERAL, "<-x-> %d" % placeInQueue)
                    if not placeInQueue % 10000:
                       lcpleQueue.put((placeInQueue, None))
                    lcpleQueue.put((wId, context))
                    placeInQueue += 1

            self.condPrint(PrintLevel.GENERAL, "Putting an end to this shit!")
            for _ in range(self.threads):
                lcpleQueue.put((None, None))
            
            for process in processes:
                process.join()

            self.sumLSAConfPLEPerContext = lcpleD

            self.condPrint(PrintLevel.NORMAL, " <<< Done with parallel processing of PLEs")
            

        if self.writeContextLSAConfPLEsFile and not self.readContextLSAConfPLEsFile:
            self.condPrint(PrintLevel.STEPS, ">  Writing Context LC PLs to file")
            cWriteStart = time.time()
            cpFile = open(self.writeContextLSAConfPLEsFile, 'wb')
            pickle.dump(self.sumLSAConfPLEPerContext, cpFile)
            cpFile.close()
            self.condPrint(PrintLevel.TIME, " - Writing Context LC PLs took %f seconds" % (time.time() - cWriteStart))
            self.condPrint(PrintLevel.STEPS, "<  Done writing Context LC PLs to file")

        ### LC PB STUFF #################################

        self.condPrint(PrintLevel.GENERAL, "-- Processing the LSA conf PBEs")

        cpStart = time.time()
        if self.readContextLSAConfPBEsFile:
            self.condPrint(PrintLevel.STEPS, ">  Reading Context LC PBs from file")
            cpFile = open(self.readContextLSAConfPBEsFile, 'rb')
            self.sumLSAConfPLEBerContext = pickle.load(cpFile)
            cpFile.close()
        elif self.plainLSAConfPBEsFile:
            self.condPrint(PrintLevel.STEPS, ">  Reading tab-separated Context LC PBs from file")
            with open(self.plainLSAConfPBEsFile, 'r') as f:
                for line in f:
                    lContext, rContext, value = line.rstrip().split('\t')
                    context = lContext + '\t' + rContext
                    self.sumLSAConfPBEPerContext[context] = float(value)
        else:
            self.condPrint(PrintLevel.STEPS, ">  Computing Context LC PBs")

            self.condPrint(PrintLevel.NORMAL, " >>> Starting parallel processing of PBEs")

            lcpbeManager = Manager()
            lcpbeD = lcpbeManager.dict()
            lcpbeL = lcpbeManager.Lock()
            lcpbeQueue = Queue()
            processes = []

            for i in range(self.threads):
                process = Process(target=self.parallelLCPBE, args=(i,lcpbeQueue,lcpbeD,lcpbeL,))
                processes.append(process)
                process.start()

            placeInQueue = 0
            for contextId, context in enumerate(self.contextCentroids):
                for wId in self.id2word:
                    #self.condPrint(PrintLevel.GENERAL, "<-x-> %d" % placeInQueue)
                    if not placeInQueue % 10000:
                       lcpbeQueue.put((placeInQueue, None))
                    lcpbeQueue.put((wId, context))
                    placeInQueue += 1

            self.condPrint(PrintLevel.GENERAL, "Putting an end to this shit!")
            for _ in range(self.threads):
                lcpbeQueue.put((None, None))
            
            for process in processes:
                process.join()

            self.sumLSAConfPBEPerContext = lcpbeD

            self.condPrint(PrintLevel.NORMAL, " <<< Done with parallel processing of PBEs")
            

        if self.writeContextLSAConfPBEsFile and not self.readContextLSAConfPBEsFile:
            self.condPrint(PrintLevel.STEPS, ">  Writing Context LC PBs to file")
            cWriteStart = time.time()
            cpFile = open(self.writeContextLSAConfPBEsFile, 'wb')
            pickle.dump(self.sumLSAConfPBEPerContext, cpFile)
            cpFile.close()
            self.condPrint(PrintLevel.TIME, " - Writing Context LC PBs took %f seconds" % (time.time() - cWriteStart))
            self.condPrint(PrintLevel.STEPS, "<  Done writing Context LC PBs to file")

    def close(self):
        if self.outputFile:
            self.of.close()


