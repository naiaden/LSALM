#!/usr/bin/python
# -*- coding: utf-8 -*-

# reads w1 w2 ... wn-2 f

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

class LsaLMR(LsaLM):
  
    def getContext(self, text):
        words = text.split()
        leftContext = ' '.join(words[0:-2])
        return leftContext + '\t'
   
    def getPcontext(self, text):
        words = text.split()
        leftContext = ' '.join(words[-3:-1])
        return leftContext

    def getFocusWord(self, text):
        words = text.split()
        return words.pop()
  
    def getCentroid(self, text):
        words = text.split()
        
        centroid = None
       
        tempCentroid = [0]*int(self.dimensions)
        cFilled = False
        
        for word in words:
            nTuple = self.id2word.doc2bow(word.split())
            if nTuple:
                cFilled = True
                nId = nTuple[0][0]
                nProjection = self.lsi.projection.u[nId]
                tempCentroid += nProjection

        if cFilled:
            return tempCentroid
        else:
            return centroid


lm = LsaLMR(sys.argv[1:])
lm.buildSpace()
