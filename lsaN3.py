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

            leftContext, rightContext = context.split('\t')

            PLvalue = self.PL(focusId,context)
            lc = self.getPrecachedLSAConf(focusId)
            return (PLvalue, "%s %s %s" % (leftContext, focusWord, rightContext), lc, focusId)
        return None    

    def evaluate(self):
        pass
#
#        self.condPrint(PrintLevel.GENERAL, "-- Evaluating contexts")
#
#        if self.trainFile:
#            with open(self.trainFile, 'r') as f:
#                self.condPrint(PrintLevel.SPECIFIC, "  -- Computing probability per word")
#                for line in f:
#                    line = line.rstrip()
#                    focusWord = line.split()[-1]
#                    self.condPrint(PrintLevel.SPECIFIC, "     - Processing %s" % focusWord)
#                    context = ' '.join(line.split()[0:2]) + '\t' + '' # empty right context
#
#                    if context in self.contextCentroids:
#                        result = self.evaluateWordForContext(context, focusWord)
#                        if result:
#                            (pl, text, lc, wId) = result
#
#                            wProb = p + self.getSRILMProb(wId ,context) 
#                            normalisation = 0
#                            for wId in self.id2word:
#                                normalisation += pow(self.PL(wId, context), l) + pow(self.getSRILMProb(wId, context), 1-l)
#                            outputString = "" % ()
#                            #outputString = "%.16f\t%s\t%.16f\t%.16f\t%.16f" % (p, c, l, self.sumPLEPerContext[context], self.sumLSAConfPLEPerContext[context])
#
#                            if self.outputFile:
#                                self.of.write("%s\n" % outputString)
#                            else:
#                                print(outputString)

lm = LsaLMN3(sys.argv[1:])
lm.buildSpace()
lm.evaluate()
