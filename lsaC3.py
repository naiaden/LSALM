#!/usr/bin/python
# -*- coding: utf-8 -*-

# w1 f w2

import logging
import gensim
import numpy
import math
import sys
import getopt
import time
import pickle

from lsalm import LsaLM
from lsalm import PrintLevel

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class LsaLMC3(LsaLM):
   
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
                        words = line.rstrip().split()
                        
                        leftContext = words[0]
                        rightContext = words[2]
                        context = leftContext + '\t' + rightContext

                        n1Tuple = self.id2word.doc2bow(words[0].split())
                        n3Tuple = self.id2word.doc2bow(words[2].split())
                        if n1Tuple and n3Tuple:
                            n1Id = n1Tuple[0][0]
                            n1Projection = self.lsi.projection.u[n1Id]
                            n3Id = n3Tuple[0][0]
                            n3Projection = self.lsi.projection.u[n3Id]

                            centroid = n1Projection + n3Projection
                            self.contextCentroids[context] = centroid
            else:
                condPrint(PrintLevel.STEPS, ">  No train file is given, therefore no contexts!")
        self.condPrint(PrintLevel.TIME, " - Took %f seconds" % (time.time() - readCStart))
        self.condPrint(PrintLevel.STEPS, "<  Done reading contexts")
        
        if self.writeContextsFile and not self.readContextsFile:
            self.condPrint(PrintLevel.STEPS, ">  Writing contexts to file")
            cWriteStart = time.time()
            rcFile = open(self.writeContextsFile, 'wb')
            pickle.dump(self.contextCentroids, rcFile)
            rcFile.close()
            self.condPrint(PrintLevel.TIME, " - Took %f seconds" % (time.time() - cWriteStart))
            self.condPrint(PrintLevel.STEPS, "<  Done writing contexts to file")

    def evaluateForContext(self, context):

        for focusWordId in self.id2word:
            wordStart = time.time()
            centroid = self.contextCentroids[context]
     
            focusWord = self.id2word[focusWordId]
            focusWordListForm = focusWord.split(None, 1)
            focusTuple = self.id2word.doc2bow(focusWordListForm)
    
            if focusTuple:
                focusId = focusTuple[0][0]
                focusProjection = self.lsi.projection.u[focusId]
    
                PLvalue = self.PL(focusProjection,centroid)
                lc = self.getPrecachedLSAConf(focusId)
                leftContext, rightContext = context.split('\t') 
                leftContextWords = leftContext.split()
                rightContextWords = rightContext.split()
                outputString = "%.16f\t%s %s %s\t%.16f" % (PLvalue, leftContextWords[0], focusWord, rightContextWords[0], lc)
                if self.outputFile:
                    self.of.write(outputString + "\n")
                else:
                    print(outputString)
            self.condPrint(PrintLevel.EVERYTHING, "        %s in %f" % (focusWord, time.time() - wordStart))

    def evaluate(self):
        self.readContexts()

        self.condPrint(PrintLevel.GENERAL, "-- Evaluating contexts")

        if self.evaluatePart:
            if self.evaluatePart > 0:
                evaluateF = math.floor(len(self.contextCentroids)*1.0*(self.evaluatePart-1)/self.taskParts)
                evaluateT = math.floor(len(self.contextCentroids)*1.0*(self.evaluatePart)/self.taskParts)
            else:
                evaluateF = 0
                evaluateT = len(self.contextCentroids)

            partStart = time.time()
            partNumber = 0
            for contextId,context in enumerate(self.contextCentroids):
                if evaluateF < contextId <= evaluateT:
                    self.condPrint(PrintLevel.SPECIFIC, "   > Starting context %d: %s" % (contextId, context))
                    cStart = time.time()
                    self.evaluateForContext(context) 
                    cDelta = time.time() - cStart
                    self.condPrint(PrintLevel.SPECIFIC, "   - Took %f seconds (avg: %f)" % (cDelta, cDelta/len(self.id2word)))
                    self.condPrint(PrintLevel.SPECIFIC, "   < Done with context %d: %s" % (contextId, context)) 

                    partNumber += 1
                    if not partNumber % 100:
                        self.condPrint(PrintLevel.TIME, "   | part %d: avg %f" % (partNumber, (time.time() - partStart)/100))
                        partStart = time.time()
        


lm = LsaLMC3(sys.argv[1:])
lm.buildSpace()
lm.evaluate()
