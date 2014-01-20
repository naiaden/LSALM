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
    manager = Manager()
    printLock = manager.Lock()

#   id2word
#   mm
#   lsi
    corpusCounts = {}     # contains the corpus count of a word id
    LSAConfCacheNoms = {} # contains the lsa confidence value of a word id
    pcontextIndex = {}    # contains the plain contexts
    contextIndex = {}     # contains the contexts
    contextPcontext ={}  # contains the link between cId and pcId
    contextFocusWords = {}# contains the wId's of focus words in the cId

    mcQ = Queue()
    mcProcesses = []

    pleQ = Queue()
    pleProcesses = []
    
    verbosity = int(0)
    programIdentifier = ''
    threads = 24
    
    distributed = False
    mmfile = 'markovmarket'
    dictfile = 'dictionary'
    normfile = ''
    gamma = float(7.0)
    dimensions = 150
    testFile = ''

    srilmProbsDirectory = ''
    readContextIndexFile = ''

    writeLSAFile = ''
    readLSAFile = ''
    outputFile = ''
    readWordCountFile = ''
    writeWordCountFile = ''
    readLSAConfFile = ''
    writeLSAConfFile = ''  
 
    def printHelp(self):
        self.condPrint(PrintLevel.NORMAL, "Nothing to see here at the moment. Look inside the file")

    def cos(self,wordId,contextCentroid):
        w = self.lsi.projection.u[wordId]

        result = 0

        if contextCentroid is not None:
            nom = numpy.dot(w,contextCentroid)
            det1 = numpy.linalg.norm(w)
            det2 = numpy.linalg.norm(contextCentroid)
            if math.isnan(nom) or math.isnan(det1) or math.isnan(det2) or det1 == 0 or det2 == 0 or nom == 0:
                return 0
            elif det1*det2 <= 0:
                raise InvalidValueError(Fore.RED + "The normalisation of the cos function shouldn't be <= 0!\nValue: %f\nWord: %s\nContext: %s" % (det1*det2, self.id2word[wordId], context))
            # FORMULA (2)
            else:
                val = nom/(det1*det2)
                if not math.isnan(val):
                    result = val
        
        return result
    
    def minCos(self,contextCentroid):
        minVal = 0
        minId = 0
        cosSum = 0
        for wId in self.id2word:
            val = self.cos(wId, contextCentroid)
            cosSum += val
            # FORMULA (3)
            if(val < minVal):
                minVal = val
                minId = wId
            
        plDen = cosSum - len(self.id2word) * minVal
        return (minId, minVal, cosSum, plDen)


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
                    self.LSAConfCacheNoms[wId] = self.LSAConfCacheNoms.get(wId, 0) + Pij * math.log(Pij)
        for keyId, key in enumerate(self.LSAConfCacheNoms):
            self.LSAConfCacheNoms[key] = 1+ self.LSAConfCacheNoms[key]/math.log(self.mm.num_docs)
    
    def getPrecachedLSAConf(self,wId):
        # LSAConfCacheNoms starts with 0
        # The rest starts with 1
        return self.LSAConfCacheNoms.get(wId-1, 0)

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
            self.printLock.acquire()
            print prefix + text
            self.printLock.release()

    def __init__(self, cmdArgs):
        try:
            opts, args = getopt.getopt(cmdArgs, 'hI:T:Dm:d:v:r:w:c:C:l:L:s:S:i:o:g:k:', ['help', 'id=', 'threads=', 'distributed', 'mfile=', 'dfile=', 'gamma=', 'dimensions=', 'test=', 'read=', 'write=', 'save=', 'verbosity=', 'readcount=', 'writecount=', 'readlsaconf=', 'writelsaconf=', 'srilm=', 'contextindex='])
        except getopt.GetoptError:
            self.printHelp()
            sys.exit(2)
        
        for (opt, arg) in opts:
            if opt in ('-h', '--help'):
                self.verbosity = (5)
                self.printHelp()
                sys.exit()
            if opt in ('-I', '--id'):
                self.programIdentifier = arg
            if opt in ('-T', '--threads'):
                self.threads = int(arg)
            elif opt in('-D', '--distributed'):
                self.distributed = True
            elif opt in ('-m', '--mfile'):
                self.mmfile = arg
            elif opt in ('-d', '--dfile'):
                self.dictfile = arg               
            elif opt in ('-v', '--verbosity'):
                self.verbosity = int(arg)               

            elif opt in ('-r', '--read'):
                self.readLSAFile = arg
            elif opt in ('-w', '--write'):
                self.writeLSAFile = arg

            elif opt in ('-c', '--readcount'):
                self.readWordCountFile = arg
            elif opt in ('-C', '--writecount'):
                self.writeWordCountFile = arg
                
            elif opt in ('-l', '--readlsaconf'):
                self.readLSAConfFile = arg
            elif opt in ('-L', '--writelsaconf'):
                self.writeLSAConfFile = arg                

            elif opt in ('-s', '--contextindex'):
                self.readContextIndexFile = arg
            elif opt in ('-S', '--srilm'):
                self.srilmProbsDirectory = arg
                
            elif opt in ('-i', '--test'):
                self.testFile = arg                              
            elif opt in ('-o', '--save'):
                self.outputFile = arg               
                
            elif opt in ('-g', '--gamma'):
                self.gamma = float(arg)
            elif opt in ('-k', '--dimensions'):
                self.dimensions = arg
       
        init(autoreset=True)

        self.condPrint(PrintLevel.GENERAL, "Program identifier: %s" % self.programIdentifier)
        self.condPrint(PrintLevel.GENERAL, "Number of threads: %d" % self.threads)
        self.condPrint(PrintLevel.GENERAL, "Corpus file: %s" % self.mmfile)
        self.condPrint(PrintLevel.GENERAL, "Distributed: %s" % ("Yes" if self.distributed else "No"))
        self.condPrint(PrintLevel.GENERAL, "Dictionary file: %s" % self.dictfile)
        
        self.condPrint(PrintLevel.GENERAL, "Verbosity level: %s" % self.verbosity)
        
        self.condPrint(PrintLevel.GENERAL, "Save LSA in: %s" % self.writeLSAFile)
        self.condPrint(PrintLevel.GENERAL, "Read LSA from: %s" % self.readLSAFile)
        
        self.condPrint(PrintLevel.GENERAL, "Read word counts from: %s" % self.readWordCountFile)
        self.condPrint(PrintLevel.GENERAL, "Write word counts to: %s" % self.writeWordCountFile)
        
        self.condPrint(PrintLevel.GENERAL, "Read LSA confidence values from: %s" % self.readLSAConfFile)
        self.condPrint(PrintLevel.GENERAL, "Write LSA confidence values to: %s" % self.writeLSAConfFile)
        
        self.condPrint(PrintLevel.GENERAL, "Gamma: %f" % self.gamma)
        self.condPrint(PrintLevel.GENERAL, "Dimensions: %s" % self.dimensions)
        
        self.condPrint(PrintLevel.GENERAL, "Context index: %s" % self.readContextIndexFile)
        self.condPrint(PrintLevel.GENERAL, "SRILM n-gram probabilities in: %s" % self.srilmProbsDirectory)
        
        self.condPrint(PrintLevel.GENERAL, "Evaluate on: %s" % self.testFile)
        self.condPrint(PrintLevel.GENERAL, "Write output to: %s" % self.outputFile)
        
        

    def writeToFile(self, queue, ):
        if self.outputFile:
            fh = open(self.outputFile, 'w')
            while True:
                text = queue.get()
                if not text:
                    fh.close()
                    return
            
                fh.write("%s\n" % text)
        else:
            while True:
                text = queue.get()
                if not text:
                    return       
                print text
            
    def processContext(self, pId, queue, queueOut):
        while True:
            context = queue.get()
            if not context:
                return       
  
            self.condPrint(PrintLevel.GENERAL, "-----")

            self.condPrint(PrintLevel.GENERAL, "   -- [%d] context: %s" % (pId, context))

            cIdx = self.contextIndex.get(context, None) 
            
            self.condPrint(PrintLevel.GENERAL, "   -- [%d] cIdx (%s)" % (pId, cIdx))
            
            pcontext = self.getPcontext(context)

            self.condPrint(PrintLevel.GENERAL, "   -- [%d] pcontext: %s" % (pId, pcontext))

            pIdx = self.pcontextIndex.get(pcontext, None) 
            
            self.condPrint(PrintLevel.GENERAL, "   -- [%d] pIdx (%s)" % (pId, pIdx))
 
            focusWords = self.contextFocusWords.get(cIdx, [])
            if cIdx is not None and pIdx is not None and len(focusWords) > 0:
                self.condPrint(PrintLevel.GENERAL, "   -- [%d] Processing context (%d) %s" % (pId, cIdx, context))

                cIdx = int(cIdx)
                pIdx = int(pIdx)

                ### PL PART ###################################

                contextCentroid = self.getCentroid(context)
                (minId, minVal, cosSum, plDen) = self.minCos(contextCentroid)          
            
                PLestCache = {}
                sumPLest = 0
            
                for wId in self.id2word:                 
                    wordCos = self.cos(wId, contextCentroid)
                    PLest = 0
                    if plDen:
                        PLest = (wordCos - minVal) / plDen
                    PLestCache[wId] = PLest
                    #self.condPrint(PrintLevel.GENERAL, "%.16f contributed by: %s" % (PLest, self.id2word[wId]))
                    sumPLest += PLest

                #self.condPrint(PrintLevel.GENERAL, "Which sums up to %.16f sumPLest" % (sumPLest))

                PLCache = {}
            
                for wId in self.id2word:
                    PL = 0
                    if PLestCache.get(wId, None) is not None:
                        PL = pow(PLestCache[wId], self.gamma) / pow(sumPLest, self.gamma)
                    #self.condPrint(PrintLevel.GENERAL, "%.16f -> %.16f by: %s" % (PLestCache[wId], PL, self.id2word[wId]))    
                    PLCache[wId] = PL
            
                ### PB PART ###################################
          
                PBCache = {}
                sumPB = 0
                
                with open("%s/context.%d" % (self.srilmProbsDirectory, pIdx), 'r') as f:
                    for line in f:
                        text, logprob = line.rstrip().split('\t')
                        
                        focusWord = self.getFocusWord(text)
                        
                        logprob = float(logprob)
                        
                        PB = math.exp(logprob)                            
                        PBCache[focusWord] = PB     
                        sumPB += PB                                      
        
                sumPLPB = 0
        
                for wId in self.id2word:
                    lc = self.getPrecachedLSAConf(wId)
                    word = self.id2word[wId]
                    PL = PLCache[wId]
                    PB =  PBCache[word]
                    sumPLPB += pow(PL, lc)*pow(PB, 1-lc)
        
                for fwId in focusWords:
                    lc = self.getPrecachedLSAConf(fwId)
                    
                    word = self.id2word[fwId]
                    PL = PLCache[fwId]
                    PB =  PBCache[word]
                    
                    P = 0
                    if sumPLPB:
                        P = pow(PL, lc) * pow(PB, 1-lc) / sumPLPB
                    
                    #print fwId, self.id2word[fwId], P
                    text = self.getText(context, word)
                    queueOut.put("%s\t%.16f\t%.16f\t%.16f\t%.16f" % (text, P, PL, PB, lc))
            else:
                pass
                #self.condPrint(PrintLevel.GENERAL, "   -- [%d] Not processing context %s because it has no focus words" % (pId, context))

    def buildSpace(self):

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
        if not self.readLSAFile:
            self.condPrint(PrintLevel.STEPS, ">  Generating LSA model")
            self.lsi = gensim.models.lsimodel.LsiModel(corpus=self.mm, id2word=self.id2word, num_topics=self.dimensions, distributed=self.distributed)
        else:
            self.condPrint(PrintLevel.STEPS, ">  Reading LSA model")
            self.lsi = gensim.models.lsimodel.LsiModel.load(self.readLSAFile)
        self.condPrint(PrintLevel.TIME, " - Took %f seconds" % (time.time() - lsiStart))
        self.condPrint(PrintLevel.STEPS, "<  Done processing the LSA model")
        
        if self.writeLSAFile and not self.readLSAFile:
            self.condPrint(PrintLevel.STEPS, ">  Writing LSA model to file")
            lsaWStart = time.time()
            self.lsi.save(self.writeLSAFile)
            self.condPrint(PrintLevel.TIME, " - Took %f seconds"  % (time.time() - lsaWStart))
            self.condPrint(PrintLevel.STEPS, "<  Done writing LSA model to file: %s" % self.writeLSAFile)

        ### Word Counts #################################
        
        self.condPrint(PrintLevel.GENERAL, "-- Processing Word Counts")

        countStart = time.time()
        if not self.readWordCountFile:
            self.condPrint(PrintLevel.STEPS, ">  Counting words")
            self.getWordCounts()
        else:
            self.condPrint(PrintLevel.STEPS, ">  Reading word counts")
            wcFile = open(self.readWordCountFile, 'rb')
            self.corpusCounts = pickle.load(wcFile)
            wcFile.close()
        self.condPrint(PrintLevel.TIME, " - Reading %d words took %f seconds" % (len(self.corpusCounts), time.time() - countStart))
        self.condPrint(PrintLevel.STEPS, "<  Done counting words")
        
        if self.writeWordCountFile and not self.readWordCountFile:
            self.condPrint(PrintLevel.STEPS, ">  Writing word counts to file")
            wcWriteStart = time.time()
            wcFile = open(self.writeWordCountFile, 'wb')
            pickle.dump(self.corpusCounts, wcFile)
            wcFile.close()
            self.condPrint(PrintLevel.TIME, " - Writing %d words took %f seconds" % (len(self.corpusCounts), time.time() - wcWriteStart))
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
            self.LSAConfCacheNoms = pickle.load(lcFile)
            wcFile.close()
        self.condPrint(PrintLevel.TIME, " - Processing %d LSA confs took %f seconds" % (len(self.LSAConfCacheNoms), time.time() - lsaconfStart))
        self.condPrint(PrintLevel.STEPS, "<  Done reading LSA confs")
        
        if self.writeLSAConfFile and not self.readLSAConfFile:
            self.condPrint(PrintLevel.STEPS, ">  Writing LSA confidence counts to file")
            lcWriteStart = time.time()
            lcFile = open(self.writeLSAConfFile, 'wb')
            pickle.dump(self.LSAConfCacheNoms, lcFile)
            lcFile.close()
            self.condPrint(PrintLevel.TIME, " - Writing %d LSA confs took %f seconds" % (len(self.LSAConfCacheNoms), time.time() - lcWriteStart))
            self.condPrint(PrintLevel.STEPS, "<  Done writing to file")

        ### Read Pcontexts ###############################
        
        self.condPrint(PrintLevel.GENERAL, "-- Reading pcontexts") # created with createContexts
        
        rcStart = time.time()
        if self.readContextIndexFile:
            with open(self.readContextIndexFile, 'r') as f:
                for line in f:
                    e = line.rstrip().split()
                    idx = e[0]
                    pctx = e[1:]
                    self.pcontextIndex[' '.join(pctx)] = idx
        else:
            self.condPrint(PrintLevel.NORMAL, Fore.RED + "You have to provide a pcontext index at this stage")
            sys.exit(3)
        self.condPrint(PrintLevel.TIME, " - Reading %d pcontexts took %f seconds" % (len(self.pcontextIndex), time.time() - lsaconfStart))
        self.condPrint(PrintLevel.STEPS, "<  Done reading pcontexts")

        ### Read Contexts ###############################
        
        self.condPrint(PrintLevel.GENERAL, "-- Reading contexts") # created with createContexts
        
        rcStart = time.time()
        if self.readContextIndexFile:
            with open(self.testFile, 'r') as f:
                cidx = 0
                for line in f:
                    text = line.rstrip()
                    ctx = self.getContext(text)
                    if ctx not in self.contextIndex:  
                        self.contextIndex[ctx] = cidx
                        cidx += 1
                        
                        pctx = ctx.replace('\t', ' ').rstrip()
                        pidx = self.pcontextIndex.get(pctx, None)
                        if pidx is not None:
                            self.contextPcontext[cidx] = pidx
 
                    fw = self.getFocusWord(text)
                    idx = self.contextIndex[ctx]
                    a = self.contextFocusWords.get(idx, [])
                    
                    fwTuple = self.id2word.doc2bow(fw.split())
                    if fwTuple:
                        a.append(fwTuple[0][0])
                        self.contextFocusWords[idx] = a  
                    
      
        else:
            self.condPrint(PrintLevel.NORMAL, Fore.RED + "You have to provide a train file at this stage")
            sys.exit(3)
        self.condPrint(PrintLevel.TIME, " - Reading %d contexts took %f seconds" % (len(self.contextIndex), time.time() - lsaconfStart))
        self.condPrint(PrintLevel.STEPS, "<  Done reading contexts")

        ### Process Contexts ############################

        self.condPrint(PrintLevel.GENERAL, "-- Processing contexts (in parallel)")
        
        cpStart = time.time()
        
        contextQueue = Queue()
        contextProcesses = []
        
        writeQueue = Queue()
        writeProcess = Process(target=self.writeToFile, args=(writeQueue,))
        writeProcess.start()

        for i in range(self.threads):
            process = Process(target=self.processContext, args=(i,contextQueue,writeQueue))
            contextProcesses.append(process)
            process.start()
        
        if self.testFile:
            for context in self.contextIndex.keys():
                 contextQueue.put(context)
                
        for _ in contextProcesses:
            contextQueue.put(None)

        for process in contextProcesses:
            process.join()
            
        writeQueue.put(None)
        writeProcess.join()

        self.condPrint(PrintLevel.TIME, " - Processing contexts took %f seconds" % (time.time() - cpStart))
        self.condPrint(PrintLevel.STEPS, "<  Done processing contexts")


