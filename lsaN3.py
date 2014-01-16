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
  
    def getText(self, context, word):
        leftContext, rightContext = context.split('\t')
        return "%s %s %s" % (leftContext, word, rightContext)
  
    def getContext(self, text):
        words = text.split()
        leftContext = ' '.join(words[0:2])
        rightContext = ''
        return leftContext + '\t' + rightContext
    
    def getFocusWord(self, text):
        words = text.split()
        return words[2]
  
    def getCentroid(self, text):
        words = text.split()
        
        centroid = None
        
        n1Tuple = self.id2word.doc2bow(words[0].split())
        n2Tuple = self.id2word.doc2bow(words[1].split())
        if n1Tuple and n2Tuple:
            n1Id = n1Tuple[0][0]
            n1Projection = self.lsi.projection.u[n1Id]
            n2Id = n2Tuple[0][0]
            n2Projection = self.lsi.projection.u[n2Id]

            centroid = n1Projection + n2Projection
        return centroid


lm = LsaLMN3(sys.argv[1:])
lm.buildSpace()
