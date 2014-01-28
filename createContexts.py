import sys
import getopt
from threading import Thread
from multiprocessing import Process, Pool, Queue

contextsFile = 'dev-N3.cxp'
contextsIndex = 'N3C.index'
contextsDir = 'N3C'
contexts = {}

vocabularyFile = '20k.vocab'
vocabulary = []


def printNumber(pId, queue):
    while True:
        l, r, n = queue.get()
        if not l and not r and not n:
            return
        #print "Processing %d" % n
        with open("%s/%s.%d" % (contextsDir, "context", n), 'w') as f:
            for word in vocabulary:
                f.write("%s %s %s\n" % (l, word, r))

try:
    opts, args = getopt.getopt(sys.argv[1:], 'c:v:I:C:', ['contextfile=', 'vocabulary=', 'index=', 'contextdir='])
except getopt.GetoptError:
    print "Invalid arguments"
    sys.exit(2)

for (opt, arg) in opts:
    if opt in ('-c', '--contextfile'):
        contextsFile = arg
    elif opt in ('-v', '--vocabulary'):
        vocabularyFile = arg
    elif opt in ('-I', '--index'):
        contextsIndex = arg
    elif opt in ('-C', '--contextdir'):
        contextsDir = arg

print "contexts: %s" % contextsFile
print "index: %s" % contextsIndex
print "c-dir: %s" % contextsDir
print "vocabulary: %s" % vocabularyFile

with open(vocabularyFile, 'r') as f:
    for line in f:
        vocabulary.append(line.rstrip())


q = Queue()

processes = []
for i in range(30):
    process = Process(target=printNumber, args=(i, q,))
    processes.append(process)
    process.start()

d = 0

with open(contextsFile, 'r') as f:
    for lineNr, line in enumerate(f):
        context = line.rstrip()
        if context not in contexts:
            contexts[context] = lineNr
            
        rContext = ''
        cs = context.split('\t')
        lContext = cs[0]
        if len(cs) > 1:
            rContext = cs[1]
        
        q.put((lContext, rContext, lineNr))
        d = lineNr

print d

for _ in processes:
    q.put((None, None, None))

for process in processes:
    process.join()  

print "Done"


ci = open(contextsIndex, 'w')
for k,v in contexts.iteritems():
    ci.write("%d %s\n" % (v,k))
ci.close()
