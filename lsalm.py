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

class PrintLevel:
    (NORMAL, GENERAL, STEPS, TIME, SPECIFIC, EVERYTHING) = range(0, 6)

class LsaLM:

    confidenceCache = {}
    minCosCache = {}
    corpusCounts = {}
    LSAconfCacheNoms = {}
    contextCentroids = {}
    normalisationCache = {}
    
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
        print("-t, --train f              read lines to apply trained model on")
        print("-w, --write f              write LSA model to file f")
        print("-r, --read f               read LSA from file f")
        print("-s, --save f               save output (probability and lsa confidence) to file f")
        print("-c, --readcount f          read word count file from f")
        print("-C, --writecount f         write word count to file f")
        print("-l, --readlsaconf f        read lsa confidence values from f")
        print("-L, --writelsaconf f       write lsa condifence values to f")
        print("-n, --readnormalisation f  read normalisation cache from f")
        print("-N, --writenormalisation f write normalisation cache to f")
        print("-T, --thousand             divide task in 1000 pieces, rather than the default 100")
        print("-e, --evaluatepart n       evaluate subset n (of 100, unless -T), all=-1 (default=-1)")
        print("-v, --verbosity n          set verbosity level (default=0)")    
        print("                           (0=normal, 1=general, 2=steps, 3=time, 4=specific, 5=everything)")
    
    
    # The cosine is computed between the LSA vector for word w and the centroid C
    @staticmethod
    def cos(w,C):
        nom = numpy.dot(w,C)
        det1 = numpy.linalg.norm(w)
        det2 = numpy.linalg.norm(C)
        if math.isnan(nom) or math.isnan(det1) or math.isnan(det2) or det1 == 0 or det2 == 0 or nom == 0:
            return 0
        val = nom/(det1*det2)
        return 0 if math.isnan(val) else val
    
    # Finds the smallest cosine between the context C and anyword W_j which ranges over the N vocabulary items
    # Returns a triple with the id of the smallest cosine distance, the smallest cosine distance, and the sum of all distances
    def MinCos(self,C):
        minVal = 0
        minId = 0
        cosSum = 0
        for id in self.id2word:
            val = LsaLM.cos(self.lsi.projection.u[id], C)
            cosSum = cosSum + val
            if(val < minVal):
                minVal = val
                minId = id
        plDen = pow((cosSum - minVal), self.gamma)
        return (minId, minVal, cosSum, plDen)
    
    def getCachedMinCos(self,C):
        Cstring = C.tostring()
        if Cstring in self.minCosCache:
            return self.minCosCache[Cstring]
        else:
            mincos = self.MinCos(C)
            self.minCosCache[Cstring] = mincos
            return mincos

    def getCachedNormalisation(self,C):
        Cstring = C.tostring()
        if Cstring in self.normalisationCache:
            return self.normalisationCache[Cstring]
        else:
            normalisation = 0
            for word in self.id2word:
                normalisation += LsaLM.cos(self.lsi.projection.u[word],C)
            self.normalisationCache[Cstring] = normalisation
            return normalisation
    
    # Compresses PLest and PL, since MinCos doesn't have to be computed each time in PL
    def PL(self,iProjection,C):
        (minId, minVal, cosSum, plDen) = self.getCachedMinCos(C)
    
        nom = (LsaLM.cos(iProjection,C) - minVal) / plDen
        den = self.getCachedNormalisation(C) - len(self.id2word)*minVal;
        return nom/(den/plDen)
    
    def getWordCountById(self,id,corpus):
        if id in self.corpusCounts:
            return self.corpusCounts[id]
        else:
            return 0

    def getWordCount(self,i,corpus):
        if i-1 in self.corpusCounts:
            return self.corpusCounts[i-1]
        else:
            return 0
    
    def LSAconf(self,i,corpus):
        nom = 0;
        countOfi = self,getWordCount(i,corpus)
        for doc in corpus:
    	    for id, val in doc:
                if(i == id+1):
                    Pij = val/countOfi
                    nom = nom + Pij * math.log(Pij)
        return 1 + nom/math.log(corpus.num_docs)
    
    def buildLSAConfCache(self,corpus):
        for doc in corpus:
            for id,val in doc:
                gwc = self.getWordCountById(id, corpus)
                if gwc:
                    Pij = val/gwc
                    if id in self.LSAconfCacheNoms:
                        self.LSAconfCacheNoms[id] += Pij * math.log(Pij)
                    else:
                        self.LSAconfCacheNoms[id] = Pij * math.log(Pij)
#                else:
#                     print 
#                    self.LSAconfCacheNoms[id] 
        for key in self.LSAconfCacheNoms:
            self.LSAconfCacheNoms[key] = 1+ self.LSAconfCacheNoms[key]/math.log(corpus.num_docs)
    
    def getPrecachedLSAConf(self,id):
        if id in self.LSAconfCacheNoms:
            return self.LSAconfCacheNoms[id]
        else:
            return 0
    
    def getCachedLSAConf(self,w,i,corpus):
        if w in self.confidenceCache:
            return self.confidenceCache[w]
        else:
            conf = self.LSAconf(i,corpus)
            self.confidenceCache[w] = conf
            return conf
    
    def getWordCounts(self,corpus):
        for doc in corpus:
            for id, val in doc:
                if id in self.corpusCounts:
                    self.corpusCounts[id] = self.corpusCounts[id] + val
                else:
                    self.corpusCounts[id] = val
    
    def condPrint(self,level,text,name=''):
        if level <= self.verbosity:
            if self.programIdentifier:
                print '%s %10s %s' % (strftime("%Y%m%d %H:%M:%S", gmtime()), self.programIdentifier, text)
            else:
                print '%s %10s %s' % (strftime("%Y%m%d %H:%M:%S", gmtime()), name, text)

    def __init__(self, cmdArgs):
        try:
            opts, args = getopt.getopt(cmdArgs, 'hi:pm:x:X:d:g:k:t:w:r:s:v:c:C:l:L:n:N:Te:', ['help', 'id=', 'distributed', 'mfile=', 'readcontexts=', 'writecontexts', 'dfile=', 'gamma=', 'dimensions=','train=', 'write=', 'read=', 'save=', 'verbosity=', 'readcount=', 'writecount=', 'readlsaconf=', 'writelsaconf=', 'readnormalisation=', 'writenormalisation=', 'thousand', 'evaluatepart=' ])
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
            elif opt in ('-n', '--readnormalisation'):
                self.readNormalisationFile = arg
            elif opt in ('-N', '--writenormalisation'):
                self.writeNormalisationFile = arg
            elif opt in ('-e', '--evaluatepart'):
                self.evaluatePart = int(arg)
            elif opt in ('-T', '--thousand'):
                self.taskParts = 1000
            elif opt in ('-L', '--writelsaconf'):
                self.writeLSAConfFile = arg
        
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
        self.condPrint(PrintLevel.GENERAL, "Read normalisation factors from: %s" % self.readNormalisationFile)
        self.condPrint(PrintLevel.GENERAL, "Write normalisation factors to: %s" % self.writeNormalisationFile)
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
            self.condPrint(PrintLevel.STEPS, ">  Generating LSI model")
            self.lsi = gensim.models.lsimodel.LsiModel(corpus=self.mm, id2word=self.id2word, num_topics=self.dimensions, distributed=self.distributed)
        else:
            self.condPrint(PrintLevel.STEPS, ">  Reading LSI model")
            self.lsi = gensim.models.lsimodel.LsiModel.load(self.readLSIFile)
        self.condPrint(PrintLevel.TIME, " - Took %f seconds" % (time.time() - lsiStart))
        self.condPrint(PrintLevel.STEPS, "<  Done processing the LSI model")
        
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
            self.getWordCounts(self.mm)
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
            self.buildLSAConfCache(self.mm)
        else:
            self.condPrint(PrintLevel.STEPS, ">  Reading LSA confs")
            lcFile = open(self.readLSAConfFile, 'rb')
            self.LSAconfCacheNoms = pickle.load(lcFile)
            wcFile.close()
        self.condPrint(PrintLevel.TIME, " - Took %f seconds" % (time.time() - lsaconfStart))
        self.condPrint(PrintLevel.STEPS, "<  Done reading LSA confs")
        
        if self.writeLSAConfFile and not self.readLSAConfFile:
            self.condPrint(PrintLevel.STEPS, ">  Writing LSA confidence counts to file")
            lcWriteStart = time.time()
            lcFile = open(self.writeLSAConfFile, 'wb')
            pickle.dump(self.LSAconfCacheNoms, lcFile)
            lcFile.close()
            self.condPrint(PrintLevel.TIME, " - Took %f seconds" % (time.time() - lcWriteStart))
            self.condPrint(PrintLevel.STEPS, "<  Done writing to file")

    def close(self):
        if self.outputFile:
            self.of.close()


