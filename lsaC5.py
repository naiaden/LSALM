#!/usr/bin/python
# -*- coding: utf-8 -*-

# reads two words left-context, focus, and two words right-context trigrams

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
    
    def evaluate(self):
        ### Evaluation ##################################
        
        start = time.time()
        iteration = 0
        if self.trainFile:
            with open(self.trainFile, 'r') as f:
                for line in f:
            
                    line = line.rstrip()        
            
                    if not iteration % 10:
                        end = time.time()
                        print(">> %d iterations: %.2f s on average") % (iteration, (end-start))
                        start = time.time()
                    iteration+=1
            
                    self.condPrint(4, "-- Processing input line: %s" % line)
                    iterationStart = time.time()
            
                    self.condPrint(5, "  > Start splitting")
                    splitStart = time.time()
                    words = line.split()
                    focusWord = words[2].split(None, 1)[0]
                    focusWordListForm = focusWord.split(None, 1)
                    self.condPrint(5, "  < Done splitting in %f" % (time.time() - splitStart))
            
                    self.condPrint(5, "  > Start looking for doc2bow")
                    doc2bowStart = time.time()
                    # So the middle word from the trigram is taken,
                    # and the two other words are considered invidivually
                    n1Tuple = self.id2word.doc2bow(words[0].split())
                    n2Tuple = self.id2word.doc2bow(words[1].split())
                    n3Tuple = self.id2word.doc2bow(focusWordListForm)
                    n4Tuple = self.id2word.doc2bow(words[3].split())
                    n5Tuple = self.id2word.doc2bow(words[4].split())
                    self.condPrint(5, "  < Done looking for doc2bow in %f" % (time.time() - doc2bowStart))
            
                    if n1Tuple and n2Tuple and n3Tuple and n4Tuple and n5Tuple:
                        n1Id = n1Tuple[0][0]
                        n2Id = n2Tuple[0][0]
                        n3Id = n3Tuple[0][0]
                        n4Id = n4Tuple[0][0]
                        n5Id = n5Tuple[0][0]
                        self.condPrint(4, "  - Processing %d %d %d %d %d" % (n1Id, n2Id, n3Id, n4Id, n5Id))
            
                        self.condPrint(5, "  > Start looking for projections")
                        lookingStart = time.time()
                        n1Projection = self.lsi.projection.u[n1Id]
                        n2Projection = self.lsi.projection.u[n2Id]
                        n3Projection = self.lsi.projection.u[n3Id]
                        n4Projection = self.lsi.projection.u[n4Id]
                        n5Projection = self.lsi.projection.u[n5Id]
                        self.condPrint(5, "  < Done looking for projections in %f" % (time.time() - lookingStart))
            
                        centroid = n1Projection + n2Projection + n4Projection + n5Projection
            
                        self.condPrint(5, "  > Start computing the PL value")
                        plStart = time.time()
                        PLvalue = self.PL(n3Projection,centroid,self.lsi,self.id2word)
                        self.condPrint(5, "  < Done computing the PL value in %f" % (time.time() - plStart))
            
                        self.condPrint(5, "  > Start computing the LSAconf")
                        lcStart = time.time()
                        lc = self.getPrecachedLSAConf(n3Id)
                        self.condPrint(5, "  < Done computing the LSAconf in %f" % (time.time() - lcStart))
            
                        outputString = "%.16f\t%s %s %s %s %s\t%.16f" % (PLvalue, words[0], words[1], focusWord, words[3], words[4], lc)
                        if self.outputFile:
                            self.of.write(outputString + "\n")
                        else:
                            print(outputString)
                    else:
                        self.condPrint(4, "-- Not processing this line because one of the unigrams was not found")
                    
                    self.condPrint(4, " - Processing took %f seconds\n" % (time.time() - iterationStart))
        else:
            self.condPrint(1, "No evaluation file provided. Done!")
        
        lm.close()
        
        


lm = LsaLMC5(sys.argv[1:])
lm.buildSpace()
lm.evaluate()
