#!/usr/bin/python
# -*- coding: utf-8 -*-

# reads one word left context, focus, and one word right context

import logging
import gensim
import numpy
import math
import sys
import getopt
import time
import pickle

from lsalm import LsaLM

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class LsaLMC3(LsaLM):
   
    def readContexts(self):
        if self.readContextsFile:
            self.condPrint(2, " > Reading contexts from %s" % self.readContextsFile)
            rcFile = open(self.readContextsFile, 'rb')
            self.contextCentroids = pickle.load(rcFile)
            rcFile.close()
        else:
            if self.trainFile:
                self.condPrint(2, " > Reading contexts")
                rcStart = time.time()
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
                self.condPrint(5, " < Done reading contexts in %f" % (time.time() - rcStart))
            else:
                condPrint(1, "No train file is given, therefore no contexts!")
        if self.writeContextsFile and not self.readContextsFile:
            rcFile = open(self.writeContextsFile, 'wb')
            pickle.dump(self.contextCentroids, rcFile)
            rcFile.close()

    def evaluateForContext(self, context):

        evaluateF = math.floor(len(self.id2word)*1.0*(self.evaluatePart-1)/100)
        evaluateT = math.floor(len(self.id2word)*1.0*(self.evaluatePart)/100)

        self.condPrint(0, "Doing wordIds from %d through %d" % (evaluateF, evaluateT))
        
        for focusWordId in self.id2word:
            if evaluateF < focusWordId <= evaluateT:
                centroid = self.contextCentroids[context]
 
	        focusWord = self.id2word[focusWordId]
                focusWordListForm = focusWord.split(None, 1)
                focusTuple = self.id2word.doc2bow(focusWordListForm)

                if focusTuple:
                    focusId = focusTuple[0][0]
                    focusProjection = self.lsi.projection.u[focusId]

                    PLvalue = self.PL(focusProjection,centroid,self.lsi,self.id2word)
                    lc = self.getPrecachedLSAConf(focusId)
                    leftContext, rightContext = context.split('\t') 
                    leftContextWords = leftContext.split()
                    rightContextWords = rightContext.split()
                    outputString = "%.16f\t%s %s %s\t%.16f" % (PLvalue, leftContextWords[0], focusWord, rightContextWords[0], lc)
                    if self.outputFile:
                        self.of.write(outputString + "\n")
                    else:
                        print(outputString)

    def evaluate(self):
        self.readContexts()

        for context in self.contextCentroids:
            self.condPrint(2, "Evaluating %s" % context)
            self.evaluateForContext(context) 
        


lm = LsaLMC3(sys.argv[1:])
lm.buildSpace()
lm.evaluate()
