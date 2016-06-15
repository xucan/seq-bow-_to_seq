import pickle
import numpy as np
import rnn as rnned
import time
import sys
from util import *

def minibatch(l, bs):
    for i in xrange(0, len(l), bs):
        yield l[i:i+bs]

def saveModel(rnn):
    rParameters = rnn.getParams()
    print 'save model...'
    with open("./best.mdl","wb") as m:
        pickle.dump([rParameters], m)

# Hyperparameters
s = {
  'ifmodel':False,
  'modelpath':'./best.mdl',
  'data_path':'./data/val.txt',
  'lr': 0.1, # The learning rate
  'bs':128, # size of the mini-batch
  'nhidden':512, # Size of the hidden layer
  'seed':324, # Seed for the random number generator
  'emb_dimension':200, # The dimension of the embedding
  'nepochs':2000, # The number of epochs that training is to run for
  'vobsize':1000, # The frequency threshold for histogram pruning of the vocab
  'trainsize':1000,#The size of train samples
  'valsize':200,#The size of val samples
  'result_path':'./result.txt'
}
#load all data
tv, word_to_index = load_data(filename=s['data_path'],vocabulary_size=s['vobsize'])

#load train data
train = tv[0:s['trainsize']]
#train = tv
#load dev data
dev = tv[s['trainsize']:len(tv)]
#dev = tv[0:s['valsize']]
#dev = tv

print 'the size of trainset is: ', len(train)
print 'the size of valset is: ', len(dev)

#update the size of dict
s['vobsize'] = len(word_to_index)


if s['ifmodel']:
    print 'load model...'
    start = time.time()
    best = cPickle.load(open(s['modelpath'], "r"))
    rParameters = best[0]
    rnn = rnned.RNNED(nh=s['nhidden'], nc=s['vobsize'], de=s['emb_dimension'],model=rParameters)
    print "--- Done compiling theano functions : ", time.time() - start, "s"
else:
    start = time.time()
    rnn = rnned.RNNED(nh=s['nhidden'], nc=s['vobsize'], de=s['emb_dimension'],model=None)
    print "--- Done compiling theano functions : ", time.time() - start, "s"

s['clr'] = s['lr']
best_dev_nll = np.inf

def writeresult(result):
    f = open(s['result_path'], 'a+')
    f.write(result)
    f.write('\n')
    f.close

#Training
for e in xrange(s['nepochs']):
    s['ce'] = e
    tic = time.time()

    for i, batch in enumerate(minibatch(train, s['bs'])):
        rnn.train(batch, s['clr'])

    print '[learning] epoch', e, '>> completed in', time.time() - tic, '(sec) <<'
    result = '[learning] epoch' + str(e) + '>> completed in'+str(time.time() - tic)+'(sec) <<'
    writeresult(result)
    sys.stdout.flush()

    #get the average nll for the validation set
    dev_nlls = rnn.test(dev)
    dev_nll = np.mean(dev_nlls)
    print '[dev-nll]', dev_nll, "(NEW BEST)" if dev_nll < best_dev_nll else ""

    if dev_nll<best_dev_nll:
        result = '[dev-nll] '+ str(dev_nll)+ "(NEW BEST)"
    else:
        result = '[dev-nll] '+ str(dev_nll)
    writeresult(result)
    sys.stdout.flush()

    if dev_nll < best_dev_nll:
        best_dev_nll = dev_nll
        s['be'] = e
        saveModel(rnn)

    if abs(s['be'] - s['ce']) >= 3: s['clr'] *= 0.2
    if s['clr'] < 1e-5: break

print '[BEST DEV-NLL]', best_dev_nll
print '[FINAL-LEARNING-RATE]', s['clr']

result = '[BEST DEV-NLL]'+ str(best_dev_nll)
writeresult(result)
result = '[FINAL-LEARNING-RATE]'+ str(s['clr'])
writeresult(result)












