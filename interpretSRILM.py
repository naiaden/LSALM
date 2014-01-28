from __future__ import print_function
import sys

# Call like this: ngram -ppl <(sed 's/  //g' contextWordCartProd ) -lm train.lm -debug 1 | python ./thisscript

class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

lookupFile = 'context.lookup'
contextDir = 'contexts'

contextRef = {}

ngram = ''
phase = 1
for lineNr, line in enumerate(sys.stdin):
    line = line.rstrip()

#    if not lineNr % 1000:
#        print ("WARNING: %d" % lineNr, end='\n', file=sys.stderr)
 
    for case in switch(phase):
        if case(1): # n-gram
            ngram = ' '.join(line.split())
            phase = 2
            break
        if case(2): # n-gram probabilities
            if line.startswith('\tp('):
                prob = line.split('gram] ')[1].split(' [ ')[0]
                break
            else:
                phase = 3
        if case(3): # numbers N j K
            phase = 4
            break
        if case(4): # sentence perplexity
            phase = 0
            break
        if case(0): # empty line
            phase = 1
            output = ngram + '\t' + prob
            print(output, end='\n')
            break
   
