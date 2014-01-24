#!/usr/bin/python
# -*- coding: utf-8 -*-

# reads w1 w2 f w3 w4

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
from operator import itemgetter

class LsaLMC5(LsaLM):
  
    def getText(self, context, word):
        leftContext, rightContext = context.split('\t')
        return "%s %s %s" % (leftContext, word, rightContext)
  
    def getContext(self, text):
        words = text.split()
        leftContext = ' '.join(itemgetter(0,1)(words.split()))
        rightContext = ' '.join(itemgetter(3,4)(words.split()))
        return leftContext + '\t' + rightContext
    
    def getPcontext(self, text):
        words = text.split()
        leftContext = ' '.join(itemgetter(0,1)(words.split()))
        rightContext = ' '.join(itemgetter(3,4)(words.split()))
        return leftContext + ' ' + rightContext
    
    def getFocusWord(self, text):
        words = text.split()
        return words[2]
  
    def getCentroid(self, text):
        words = text.split()
        
        centroid = None
        
        n1Tuple = self.id2word.doc2bow(words[0].split())
        n2Tuple = self.id2word.doc2bow(words[1].split())
        n4Tuple = self.id2word.doc2bow(words[3].split())
        n5Tuple = self.id2word.doc2bow(words[4].split())
        if n1Tuple and n2Tuple and n4Tuple and n5Tuple:
            n1Projection = self.lsi.projection.u[n1Tuple[0][0]]
            n2Projection = self.lsi.projection.u[n2Tuple[0][0]]
            n4Projection = self.lsi.projection.u[n4Tuple[0][0]]
            n5Projection = self.lsi.projection.u[n5Tuple[0][0]]

            centroid = n1Projection + n2Projection + n4Projection + n5Projection
        return centroid


lm = LsaLMC5(sys.argv[1:])
lm.buildSpace()
