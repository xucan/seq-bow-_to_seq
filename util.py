from __future__ import division
from collections import Counter
import numpy as np
import cPickle
import sys
import random
import theano
import theano.tensor as T

SENTENCE_START_TOKEN = "START"
SENTENCE_END_TOKEN = "END"
UNKNOWN_TOKEN = "UNKNOWN"

def save(path, obj):
    with open(path, 'wb') as f:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)

def load_data(filename='./data/val.txt', vocabulary_size=20000, min_sent_characters=0):
    SENTENCE_START_TOKEN = "START"
    SENTENCE_END_TOKEN = "END"
    UNKNOWN_TOKEN = "UNKNOWN"

    word_to_index = []
    index_to_word = []

    word_counter = Counter()
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    sentences = []
    question = []
    for line in open(filename, 'r'):
        if line == '\n': continue
        pair = line.split('\t')
        sent = pair[1]
        que = pair[0]
        sents = sent.strip().split()
        ques = que.strip().split()
        sentences.append( [SENTENCE_START_TOKEN] + sents + [SENTENCE_END_TOKEN])
        question.append( [SENTENCE_START_TOKEN] + ques + [SENTENCE_END_TOKEN])
        word_counter.update(sents)
        word_counter.update(ques)
    
    print 'the numbers of sentences is : %d' % len(sentences)
    print 'the numbers of question is : %d' % len(question)

    vocab_count = word_counter.most_common(vocabulary_size)
    vocab = {SENTENCE_START_TOKEN:1, SENTENCE_END_TOKEN:2, UNKNOWN_TOKEN:0}
    
    for i, (word, count) in enumerate(vocab_count):
        vocab[word] = i + 3

    print 'the size of vocabulary: %d' % len(vocab)
    word_to_index = vocab
    index_to_word = {value:key for key, value in word_to_index.iteritems()}


    X = [[word_to_index.get(w,0) for w in sent[:-1]] for sent in sentences]
    Y = [[word_to_index.get(w,0) for w in sent[1:]] for sent in sentences]
    C = [[word_to_index.get(w,0) for w in sent[:-1]] for sent in question]
    C_q = []
    for que in C:
        c_train = np.zeros((len(word_to_index),)).astype('float32')
        for w in que:
            c_train[w] = 1.
        C_q.append(c_train)

    i = 0
    C_X_Y = []
    for sen in X:
        x_train = np.asarray(sen).astype('int32')
        y_train = np.asarray(Y[i]).astype('int32')
        C_X_Y.append([C_q[i], x_train, y_train])
        i = i + 1

    #save dict and trainset
    save('./data/train.pkl',C_X_Y)
    save('./data/dict.pkl', word_to_index)
    save('./data/C_q.pkl', C)

    return C_X_Y, word_to_index

def sample(a, temperature=0.5):
    a = np.array(a, dtype='double')
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a))



def print_sentence(s, index_to_word):
    sentence_str = [index_to_word[x] for x in s[1:-1]]
    print(" ".join(sentence_str))
    sys.stdout.flush()

def generate_sentence(model, index_to_word, word_to_index, c):
    # We start the sentence with the start token
    new_sentence = [word_to_index[SENTENCE_START_TOKEN]]
    while not new_sentence[-1] == word_to_index[SENTENCE_END_TOKEN]:
        next_word_probs = model.predict(c, new_sentence)[-1]
#        samples = np.random.multinomial(1, next_word_probs)
#        sampled_word = np.argmax(samples)
#        print 'sum of probs: %f' % np.sum(next_word_probs)
        sampled_word = sample(next_word_probs)
        new_sentence.append(sampled_word)

    print_sentence(new_sentence, index_to_word)
#with open('./data/small.pkl','wb') as f:
#    cPickle.dump(temp, f, protocol=cPickle.HIGHEST_PROTOCOL)


#we random select 1200 sentences from all_val.txt in which the num of trainset is 1000 and testset is 200
def process_data(filename = './data/all_val.txt', length = 1200, outpath = './data/val_NEW.txt'):

    #generate random int
    ind = []
    size = 14236
    for i in range(3000):
        temp = random.randint(0,size)
        ind.append(temp)

    index = random.sample(ind,1200)
    
    all_val = []
    for line, pairs in enumerate(open(filename,'r')):
        all_val.append(pairs)

    val = []
    for i in index:
        val.append(all_val[i])

    f = file(outpath, 'w+')
    f.writelines(val)
    f.close()

def countdictsize(filename='./data/val.txt'):
    word_counter = Counter()
    for line in open(filename,'r'):
        s = [x for x in line.strip().split()]
        word_counter.update(s)
    total_freq = sum(word_counter.values())

    print 'total freq is: ', total_freq

    print 'the number of all words is : ', len(word_counter)

    #count the number of words which freq is 1,2,3
    freq1=0
    freq2=0
    freq3=0
    freqother = 0

    vocab_count = word_counter.most_common(20000)
    print 'the number of all words is : ', len(vocab_count)
    for i, (word, count) in enumerate(vocab_count):
        if count==1:
            freq1 += 1
        elif count==2:
            freq2 += 1
        elif count==3:
            freq3 += 1
        else:
            freqother += 1
    print 'the number of freq 1 word is : ', freq1
    print 'the number of freq 2 word is : ', freq2
    print 'the number of freq 3 word is : ', freq3
    print 'the number of freq other word is: ',freqother

#load_data()
#process_data()
#countdictsize()
def rmsprop(params, gparams, learning_rate = 0.001, rho = 0.9, epsilon = 1e-6):
    updates = []
    for p, g in zip(params, gparams):
        v = p.get_value(borrow = True)
        acc = theano.shared(np.zeros(v.shape, dtype = v.dtype), broadcastable = p.broadcastable)
        acc_new = rho * acc.get_value() + (1 - rho) * g ** 2
        updates.append((acc, acc_new))
        updates.append((p, p.get_value() - learning_rate * g / T.sqrt(acc_new + epsilon).eval()))
#    save('./updates.pkl', updates)
    return updates

def adadelta(params, gparams, learning_rate = 1.0, rho = 0.95, epsilon = 1e-6):
    updates = []
    for p, g in zip(params, gparams):
        v = p.get_value(borrow = True)
        acc = theano.shared(np.zeros(v.shape, dtype = v.dtype), broadcastable = p.broadcastable)
        delta_acc = theano.shared(np.zeros(v.shape, dtype = v.dtype), broadcastable = p.broadcastable)

        acc_new = rho * acc.get_value() + (1 - rho) * g ** 2
        updates.append((acc, acc_new))

        update = (g * T.sqrt(delta_acc.get_value() + epsilon).eval() / T.sqrt(acc_new + epsilon).eval())
        updates.append((p, p.get_value() - learning_rate * update))

        delta_acc_new = rho * delta_acc.get_value() + (1 - rho) * update ** 2
        updates.append((delta_acc, delta_acc_new))
    return updates

def adagrad(params, gparams, learning_rate = 0.01, epsilon = 1e-6):
    updates = []
    for p, g in zip(params, gparams):
        v = p.get_value(borrow = True)
        acc = theano.shared(np.zeros(v.shape, dtype = v.dtype), broadcastable = p.broadcastable)
        acc_new = acc.get_value() + g ** 2
        updates.append((acc, acc_new))
        updates.append((p, p.get_value() - learning_rate * g / T.sqrt(acc_new + epsilon).eval()))
    return updates


load_data(filename='./data/val.txt', vocabulary_size=500, min_sent_characters=0)



