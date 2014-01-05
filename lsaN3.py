#!/usr/bin/python
# -*- coding: utf-8 -*-

# reads two words left-context, focus

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
from colorama import init, Fore, Back, Style

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class LsaLMN3(LsaLM):
  
    # This line has to be the core of the function:
    # self.contextCentroids[context] = centroid
    def createCentroid(self, line):
        words = line.split()
        
        leftContext = ' '.join(words[0:2])
        rightContext = ''
        context = leftContext + '\t' + rightContext

        n1Tuple = self.id2word.doc2bow(words[0].split())
        n2Tuple = self.id2word.doc2bow(words[1].split())
        if n1Tuple and n2Tuple:
            n1Id = n1Tuple[0][0]
            n1Projection = self.lsi.projection.u[n1Id]
            n2Id = n2Tuple[0][0]
            n2Projection = self.lsi.projection.u[n2Id]

            centroid = n1Projection + n2Projection
            self.contextCentroids[context] = centroid

    def evaluateWordForContext(self, context, focusWord):
        wordStart = time.time()

        focusWordListForm = focusWord.split(None, 1)
        focusTuple = self.id2word.doc2bow(focusWordListForm)

        if focusTuple:

            focusId = focusTuple[0][0]
            #self.condPrint(PrintLevel.EVERYTHING, "           * Focusword %s has id %d" % (focusWord, focusId))

            leftContext, rightContext = context.split('\t')

            PLvalue = self.PL(focusId,context)
            lc = self.getPrecachedLSAConf(focusId)
            #self.condPrint(PrintLevel.EVERYTHING, "             * %f %s %s %s %f" % (PLvalue, leftContext, focusWord, rightContext, lc))
            return (PLvalue, "%s %s %s" % (leftContext, focusWord, rightContext), lc)
        else:
            self.condPrint(PrintLevel.EVERYTHING, "          !! Word is not in vocabulary")
        self.condPrint(PrintLevel.EVERYTHING, "        %s in %f" % (focusWord, time.time() - wordStart))
        return None    

    def evaluate(self):

        self.condPrint(PrintLevel.GENERAL, "-- Evaluating contexts")

#        if self.evaluatePart:
#            if self.evaluatePart > 0:
#                evaluateF = math.floor(len(self.contextCentroids)*1.0*(self.evaluatePart-1)/self.taskParts)
#                evaluateT = math.floor(len(self.contextCentroids)*1.0*(self.evaluatePart)/self.taskParts)
#            else:
#                evaluateF = 0
#                evaluateT = len(self.contextCentroids)

        if self.trainFile:
            with open(self.trainFile, 'r') as f:


                self.condPrint(PrintLevel.SPECIFIC, "  -- Computing probability per word")
                for line in f:
                    line = line.rstrip()
                    focusWord = line.split()[-1]
                    self.condPrint(PrintLevel.SPECIFIC, "     - Processing %s" % focusWord)
                    context = ' '.join(line.split()[0:2]) + '\t' + '' # empty right context

                    if context in self.contextCentroids:    
                        self.condPrint(PrintLevel.SPECIFIC, "       -- Computing normalisation sum over context: %s" % context)
                        contextSum = 0
                        nsStart = time.time()
                        for focusWordId in self.id2word:
                            result = self.evaluateWordForContext(context, self.id2word[focusWordId]) 
                            if result:
                                (p, c, l) = result
                                contextSum += p
                        self.condPrint(PrintLevel.TIME, "          - Computing normalisation sum took %f seconds" % (time.time() - nsStart))
                        self.condPrint(PrintLevel.SPECIFIC, "            : Normalisation sum is %f" % contextSum)                  
                       
                        self.condPrint(PrintLevel.SPECIFIC, "       -- Computing word probability") 
                        wpStart = time.time()
                        result = self.evaluateWordForContext(context, focusWord)
                        self.condPrint(PrintLevel.TIME, "          - Computing word probability took %f seconds" % (time.time() - wpStart))
                        if result:
                            (p, c, l) = result
                            outputString = "%.16f\t%s\t%.16f\t%.16f" % (p, c, l, contextSum)

                            if self.outputFile:
                                self.of.write("%s\n" % outputString)
                            else:
                                print(outputString)
                    else:
                        self.condPrint(PrintLevel.EVERYTHING, "       !! No matching context found for %s" % context)

lm = LsaLMN3(sys.argv[1:])
lm.buildSpace()
#lm.readContexts()
#lm.evaluate()
