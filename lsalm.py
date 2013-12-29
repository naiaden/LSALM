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

class LsaLM:
    confidenceCache = {}
    minCosCache = {}
    corpusCounts = {}
    LSAconfCacheNoms = {}
    contextCentroids = {}
    
    verbosity = int(0)
    
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

    evaluatePart = int(0)
   
    @staticmethod 
    def printHelp():
        print("-h, --help            print this text and exit")
        print("-p, --distributed     use the distributed version of gensim")
        print("-m, --mfile f         read corpus file from f (in matrix market format)")
        print("-x, --readcontexts f  read contexts from f")
        print("-X, --writecontexts f write contexts to f")
        print("-d, --dfile f         read dictionary from f")
        print("-g, --gamma n         gamma parameter for dynamic range scaling (default=7.0)")
        print("-k, --dimensions n    the number of dimensions after SVD (default=150)")
        print("-t, --train f         read lines to apply trained model on")
        print("-w, --write f         write LSA model to file f")
        print("-r, --read f          read LSA from file f")
        print("-s, --save f          save output (probability and lsa confidence) to file f")
        print("-c, --readcount f     read word count file from f")
        print("-C, --writecount f    write word count to file f")
        print("-l, --readlsaconf f   read lsa confidence values from f")
        print("-L, --writelsaconf f  write lsa condifence values to f")
        print("-e, --evaluatepart n  evaluate subset n (of 100), all=-1 (default=-1)")
        print("-v, --verbosity n     set verbosity level (default=0)")    
    
    
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
    def MinCos(self,C,lspace,dictionary):
        minVal = 0
        minId = 0
        cosSum = 0
        for id in dictionary:
            val = LsaLM.cos(lspace.projection.u[id], C)
            cosSum = cosSum + val
            if(val < minVal):
                minVal = val
                minId = id
        plDen = pow((cosSum - minVal), self.gamma)
        return (minId, minVal, cosSum, plDen)
    
    def getCachedMinCos(self,C,lspace,dictionary):
        Cstring = C.tostring()
        if Cstring in self.minCosCache:
            return self.minCosCache[Cstring]
        else:
            mincos = self.MinCos(C,lspace,dictionary)
            self.minCosCache[Cstring] = mincos
            return mincos
    
    # Compresses PLest and PL, since MinCos doesn't have to be computed each time in PL
    def PL(self,iProjection,C,lspace,dictionary):
        (minId, minVal, cosSum, plDen) = self.getCachedMinCos(C,lspace,dictionary)
    
        nom = (LsaLM.cos(iProjection,C) - minVal) / plDen
        den = 0;
        for word in dictionary:
            den = den + (LsaLM.cos(lspace.projection.u[word],C) - minVal)
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
    
    def condPrint(self,level,text):
        if level <= self.verbosity:
            print text;

    def __init__(self, cmdArgs):
        try:
            opts, args = getopt.getopt(cmdArgs, 'hpm:x:X:d:g:k:t:w:r:s:v:c:C:l:L:e:', ['help', 'distributed', 'mfile=', 'readcontexts=', 'writecontexts', 'dfile=', 'gamma=', 'dimensions=','train=', 'write=', 'read=', 'save=', 'verbosity=', 'readcount=', 'writecount=', 'readlsaconf=', 'writelsaconf=', 'evaluatepart=' ])
        except getopt.GetoptError:
            LsaLM.printHelp()
            sys.exit(2)
        
        for (opt, arg) in opts:
            if opt in('-h', '--help'):
                LsaLM.printHelp()
                sys.exit()
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
            elif opt in ('-L', '--writelsaconf'):
                self.writeLSAConfFile = arg
        
        self.condPrint(2, "Corpus file: %s" % self.mmfile)
        self.condPrint(2, "Distributed: %s" % ("Yes" if self.distributed else "No"))
        self.condPrint(2, "Dictionary file: %s" % self.dictfile)
        self.condPrint(2, "Read contexts from: %s" % self.readContextsFile)
        self.condPrint(2, "Write contexts to: %s" % self.writeContextsFile)
        self.condPrint(2, "Read word counts from: %s" % self.readWordCountFile)
        self.condPrint(2, "Write word counts to: %s" % self.writeWordCountFile)
        self.condPrint(2, "Read LSA confidence values from: %s" % self.readLSAConfFile)
        self.condPrint(2, "Write LSA confidence values to: %s" % self.writeLSAConfFile)
        self.condPrint(2, "Gamma: %f" % self.gamma)
        self.condPrint(2, "Dimensions: %s" % self.dimensions)
        self.condPrint(2, "Evaluate on: %s" % self.trainFile)
        self.condPrint(2, "Save LSA in: %s" % self.saveLSIFile)
        self.condPrint(2, "Read LSA from: %s" % self.readLSIFile)
        self.condPrint(2, "Write output to: %s" % self.outputFile)
        self.condPrint(2, "Evaluate only part: %s" % self.evaluatePart)
        self.condPrint(2, "Verbosity level: %s" % self.verbosity)

    def buildSpace(self):
        if self.outputFile:
            self.of = open(self.outputFile, 'w')
        
        ### Dictionary and Corpus #######################
        
        self.condPrint(2, "Reading dictionary")
        self.id2word = gensim.corpora.Dictionary.load_from_text(self.dictfile)
        self.condPrint(2, self.id2word)
        
        self.condPrint(2, "Reading corpus")
        self.mm = gensim.corpora.MmCorpus(self.mmfile)
        self.condPrint(2, self.mm)
        
        ### LSA Space ###################################
        
        self.lsi = None
        lsiStart = time.time()
        if not self.readLSIFile:
            self.condPrint(2, "Generating LSI model")
            self.lsi = gensim.models.lsimodel.LsiModel(corpus=self.mm, id2word=self.id2word, num_topics=self.dimensions, distributed=self.distributed)
        else:
            self.condPrint(2, "Reading LSI model")
            self.lsi = gensim.models.lsimodel.LsiModel.load(self.readLSIFile)
        self.condPrint(2, "Done processing the LSI model in %f" % (time.time() - lsiStart))
        
        if self.saveLSIFile and not self.readLSIFile:
            self.lsi.save(self.saveLSIFile)
            self.condPrint(2, "LSI file saved to: %s" % self.saveLSIFile)
        
        ### Word Counts #################################
        
        countStart = time.time()
        self.condPrint(3, "Counting words...")
        if not self.readWordCountFile:
            self.getWordCounts(self.mm)
            self.condPrint(3, "Done counting words in %f" % (time.time() - countStart))
        else:
            wcFile = open(self.readWordCountFile, 'rb')
            corpusCounts = pickle.load(wcFile)
            wcFile.close()
            self.condPrint(3, "Done counting words from file in %f" % (time.time() - countStart))
        
        if self.writeWordCountFile and not self.readWordCountFile:
            self.condPrint(3, "Writing word counts to file")
            wcWriteStart = time.time()
            wcFile = open(self.writeWordCountFile, 'wb')
            pickle.dump(self.corpusCounts, wcFile)
            wcFile.close()
            self.condPrint(3, "Done writing to file in %f" % (time.time() - wcWriteStart))
        
        ### LSA Confidence ##############################
        
        lsaconfStart = time.time()
        self.condPrint(3, "Computing LSA confs...")
        if not self.readLSAConfFile:
            self.buildLSAConfCache(self.mm)
            self.condPrint(3, "Done computing LSA confs in %f" % (time.time() - lsaconfStart))
        else:
            lcFile = open(self.readLSAConfFile, 'rb')
            self.LSAconfCacheNoms = pickle.load(lcFile)
            wcFile.close()
            self.condPrint(3, "Done reading LSA confs from file in %f" % (time.time() - lsaconfStart))
        
        if self.writeLSAConfFile and not self.readLSAConfFile:
            self.condPrint(3, "Writing LSA confidence counts to file")
            lcWriteStart = time.time()
            lcFile = open(self.writeLSAConfFile, 'wb')
            pickle.dump(self.LSAconfCacheNoms, lcFile)
            lcFile.close()
            self.condPrint(3, "Done writing to file in %f" % (time.time() - lcWriteStart))

    def close(self):
        if self.outputFile:
            self.of.close()


