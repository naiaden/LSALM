#!/usr/bin/python
# -*- coding: utf-8 -*-

# reads w1 f w2

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

class LsaLMC3(LsaLM):
  
    def getText(self, context, word):
        leftContext, rightContext = context.split('\t')
        return "%s %s %s" % (leftContext, word, rightContext)
  
    def getContext(self, text):
        words = text.split()
        leftContext = ' '.join(itemgetter(0,2)(words.split()))
        rightContext = ''
        return leftContext + '\t' + rightContext
    
    def getPcontext(self, text):
        words = text.split()
        leftContext = ' '.join(itemgetter(0,2)(words.split()))
        rightContext = ''
        return leftContext + ' ' + rightContext
    
    def getFocusWord(self, text):
        words = text.split()
        return words[1]
  
    def getCentroid(self, text):
        words = text.split()
        
        centroid = None
        
        n1Tuple = self.id2word.doc2bow(words[0].split())
        n3Tuple = self.id2word.doc2bow(words[2].split())
        if n1Tuple and n3Tuple:
            n1Id = n3Tuple[0][0]
            n1Projection = self.lsi.projection.u[n1Id]
            n3Id = n3Tuple[0][0]
            n3Projection = self.lsi.projection.u[n3Id]

            centroid = n1Projection + n3Projection
        return centroid


lm = LsaLMC3(sys.argv[1:])
lm.buildSpace()
