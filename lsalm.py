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
    
    verbosity = int(0)
    programIdentifier = ''
    
    distributed = False
    mmfile = 'markovmarket'
    dictfile = 'dictionary'
    normfile = ''
    gamma = float(7.0)
    dimensions = 150
    trainFile = ''

    saveLSIFile = ''
    readLSIFile = ''
    outputFile = ''
    readContextPLsFile = ''
    writeContextPLsFile = ''
    readContextsFile = ''
    writeContextsFile = ''
    readWordCountFile = ''
    writeWordCountFile = ''
    readLSAConfFile = ''
    writeLSAConfFile = ''
    readNormalisationFile = ''
    writeNormalisationFile = ''

    taskParts = 100
    evaluatePart = int(0)
  
 
    @staticmethod 
    def printHelp():
        print("-h, --help                 print this text and exit")
        print("-i, --id s                 give the program an id, useful for output of parallel processes")
        print("-p, --distributed          use the distributed version of gensim")
        print("-m, --mfile f              read corpus file from f (in matrix market format)")
        print("-x, --readcontexts f       read contexts from f")
        print("-X, --writecontexts f      write contexts to f")
        print("-d, --dfile f              read dictionary from f")
        print("-g, --gamma n              gamma parameter for dynamic range scaling (default=7.0)")
        print("-k, --dimensions n         the number of dimensions after SVD (default=150)")
        print("-t, --test f               read lines to apply trained model on")
        print("-s, --save f               save output (probability and lsa confidence) to file f")

        print("-x, --readcontexts f       read contexts from f")
        print("-X, --writecontexts f      write contexts to f")
        print("-w, --write f              write LSA model to file f")
        print("-r, --read f               read LSA from file f")
        print("-c, --readcount f          read word count file from f")
        print("-C, --writecount f         write word count to file f")
        print("-l, --readlsaconf f        read lsa confidence values from f")
        print("-L, --writelsaconf f       write lsa condifence values to f")
        print("-T, --thousand             divide task in 1000 pieces, rather than the default 100")
        print("-e, --evaluatepart n       evaluate subset n (of 100, unless -T), all=-1 (default=-1)")
        print("-v, --verbosity n          set verbosity level (default=0)")    
        print("                           (0=normal, 1=general, 2=steps, 3=time, 4=specific, 5=everything)")
    
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
    
    def getCachedminCos(self,context):
        if context in self.minCosCache:
            return self.minCosCache[context]
        else:
            mincos = self.minCos(context)
            self.minCosCache[context] = mincos
            return mincos

    def getCachedPLE(self,wordId,context):
        comboString = ("%d %s" % (wordId, context))
        if comboString in self.PLEWordIdContext:
            return self.PLEWordIdContext[comboString]
        else:
            (minId, minVal, cosSum, plDen) = self.getCachedminCos(context)
            wordCos = self.cos(wordId, context)
            if wordCos < minVal:
                raise InvalidValueError(Fore.RED + "Word cosine in a context cannot be smaller than the smallest value in the context!\ncosine value (and smallest): %f (%f)\nWord: %s\nContext: %s" % (wordCos, minVal, self.id2word[wordId], context))
            # FORMULA (4)
            PLest = (wordCos - minVal) / plDen
            if PLest < 0:
                raise InvalidValueError(Fore.RED + "PLest < 0\nWord: %s\nContext: %s\nWordId lowest cosine with value:%d (%s) %f\nCossum:%f\nplDen: %f" % (self.id2word[wordId], context, minId, self.id2word[minId], minVal, cosSum, plDen))

            self.PLEWordIdContext[comboString] = PLest
            return PLest
    
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
        for key in self.LSAconfCacheNoms:
            self.LSAconfCacheNoms[key] = 1+ self.LSAconfCacheNoms[key]/math.log(self.mm.num_docs)
    
    def getPrecachedLSAConf(self,wId):
        return self.LSAconfCacheNoms.get(wId, 0)
    
    def getWordCounts(self):
        for doc in self.mm:
            for wId, val in doc:
                self.corpusCounts[wId] = self.corpusCounts.get(wId, 0) + val
   
    def readContexts(self):
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



    def condPrint(self,level,text,name=''):
        if level <= self.verbosity:
            if self.programIdentifier:
                print '%s %10s %s' % (strftime("%Y%m%d %H:%M:%S", gmtime()), self.programIdentifier, text)
            elif name:
                print '%s %10s %s' % (strftime("%Y%m%d %H:%M:%S", gmtime()), name, text)
            else:
                print '%s %s' % (strftime("%Y%m%d %H:%M:%S", gmtime()), text)

    def __init__(self, cmdArgs):
        try:
            opts, args = getopt.getopt(cmdArgs, 'hi:pm:x:X:d:g:k:t:w:r:s:v:c:C:l:L:n:N:Te:', ['help', 'id=', 'distributed', 'mfile=', 'readcontexts=', 'writecontexts', 'dfile=', 'gamma=', 'dimensions=','test=', 'write=', 'read=', 'save=', 'verbosity=', 'readcount=', 'writecount=', 'readlsaconf=', 'writelsaconf=', 'thousand', 'evaluatepart=' ])
        except getopt.GetoptError:
            LsaLM.printHelp()
            sys.exit(2)
        
        for (opt, arg) in opts:
            if opt in ('-h', '--help'):
                LsaLM.printHelp()
                sys.exit()
            if opt in ('-i', '--id'):
                self.programIdentifier = arg
            elif opt in('-p', '--distributed'):
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
       
        init(autoreset=True)

        self.condPrint(PrintLevel.GENERAL, "Program identifier: %s" % self.programIdentifier)
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
        self.condPrint(PrintLevel.GENERAL, "Save LSA in: %s" % self.saveLSIFile)
        self.condPrint(PrintLevel.GENERAL, "Read LSA from: %s" % self.readLSIFile)
        self.condPrint(PrintLevel.GENERAL, "Write output to: %s" % self.outputFile)
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
        self.condPrint(PrintLevel.TIME, self.id2word)
        self.condPrint(PrintLevel.TIME, " - Took %f seconds" % (time.time() - readDictStart))
        self.condPrint(PrintLevel.STEPS, "<  Done reading dictionary")

        
        self.condPrint(PrintLevel.STEPS, ">  Reading corpus")
        readCorpusStart = time.time()
        self.mm = gensim.corpora.MmCorpus(self.mmfile)
        self.condPrint(PrintLevel.TIME, self.mm)
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

        ### PLEs ########################################

        self.condPrint(PrintLevel.GENERAL, "-- Processing the PLEs")

        cpStart = time.time()
        if self.readContextPLsFile:
            self.condPrint(PrintLevel.STEPS, ">  Reading Context PLs from file")
            cpFile = open(self.readContextPLs, 'rb')
            self.sumPLEPerContext = pickle.load(cpFile)
            cpFile.close()
        else:
            self.condPrint(PrintLevel.STEPS, ">  Computing Context PLs")
            for context in self.contextCentroids:
                for wId in self.id2word:
                    PLEst = self.getCachedPLE(wId, context)
                    self.sumPLEPerContext[context] = self.sumPLEPerContext.get(context, 0) + pow(PLEst, self.gamma)
        self.condPrint(PrintLevel.TIME, " - Reading/computing Context PLs took %f seconds" % (time.time() - cpStart))
        self.condPrint(PrintLevel.STEPS, "<  Done reading/computing Context PLs")

        if self.writeContextPLsFile and not self.readContextPLsFile:
            self.condPrint(PrintLevel.STEPS, ">  Writing Context PLs to file")
            cWriteStart = time.time()
            cpFile = open(self.writeContextPLs, 'wb')
            pickle.dump(self.sumPLEPerContext, cpFile)
            cpFile.close()
            self.condPrint(PrintLevel.TIME, " - Writing Context PLs took %f seconds" % (time.time() - cWriteStart))
            self.condPrint(PrintLevel.STEPS, "<  Done writing Context PLs to file")

        ### Read Contexts ###############################

        self.readContexts()

    def close(self):
        if self.outputFile:
            self.of.close()


