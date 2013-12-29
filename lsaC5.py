#!/usr/bin/python
# -*- coding: utf-8 -*-

# reads two words left context, focus, and two words right context

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


class LsaLMC5(LsaLM):
   
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
                        
                        leftContext = ' '.join(words[0:2]) 
                        rightContext = ' '.join(words[3:5])
                        context = leftContext + '\t' + rightContext

                        n1Tuple = self.id2word.doc2bow(words[0].split())
                        n2Tuple = self.id2word.doc2bow(words[1].split())
                        n4Tuple = self.id2word.doc2bow(words[3].split())
                        n5Tuple = self.id2word.doc2bow(words[4].split())
                        if n1Tuple and n2Tuple and n4Tuple and n5Tuple:
                            n1Id = n1Tuple[0][0]
                            n1Projection = self.lsi.projection.u[n1Id]
                            n2Id = n2Tuple[0][0]
                            n2Projection = self.lsi.projection.u[n2Id]
                            n4Id = n4Tuple[0][0]
                            n4Projection = self.lsi.projection.u[n4Id]
                            n5Id = n5Tuple[0][0]
                            n5Projection = self.lsi.projection.u[n5Id]

                            centroid = n1Projection + n2Projection + n4Projection + n5Projection
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
                    outputString = "%.16f\t%s %s %s %s %s\t%.16f" % (PLvalue, leftContextWords[0], leftContextWords[1], focusWord, rightContextWords[0], rightContextWords[1], lc)
                    if self.outputFile:
                        self.of.write(outputString + "\n")
                    else:
                        print(outputString)

    def evaluate(self):
        self.readContexts()

        for context in self.contextCentroids:
            self.condPrint(2, "Evaluating %s" % context)
            self.evaluateForContext(context) 
        


lm = LsaLMC5(sys.argv[1:])
lm.buildSpace()
lm.evaluate()
