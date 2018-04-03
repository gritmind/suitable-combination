# -*- coding: utf-8 -*-
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import Word2Vec
from gensim.models import FastText
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from build_vocab import data_load
import pandas as pd
import numpy as np
from nltk import tokenize
import pickle
import string
import os
import re
ps = PorterStemmer()
lm = WordNetLemmatizer()
tokenizer = Tokenizer(lower=False, filters='')
global allot_PRPR_VER
global VOCA
global VOCA_gensim
global IS_GENSIM
IS_GENSIM = False

######################################################
""" LOAD & PREPROCESSING & VECTORIZING DATASET """
######################################################

def voca_clean(doc):
    global allot_PRPR_VER
    global VOCA
    global VOCA_gensim
    global IS_GENSIM
    tokens = doc.split()

    ### (1) Preprocessing for changing form/shape of tokens
    #########################################################
    lowering_vers = ['ver_a', 'ver_b', 'ver_d', 'ver_e', 'ver_f', 'ver_j', 'ver_m', 'ver_n', 'ver_p']
    if allot_PRPR_VER in lowering_vers:
        tokens = [word.lower() for word in tokens] # lower

    stemming_vers = ['ver_a', 'ver_b', 'ver_c', 'ver_e', 'ver_g', 'ver_i']
    if allot_PRPR_VER in stemming_vers:
        tokens = [ps.stem(word) for word in tokens] # stemming

    lemmatizing_vers = ['ver_m', 'ver_n', 'ver_o', 'ver_p', 'ver_q', 'ver_r']
    if allot_PRPR_VER in lemmatizing_vers:
        tokens = [lm.lemmatize(word) for word in tokens] # lemmatizing          
        
    rem_punct_vers = ['ver_a', 'ver_c', 'ver_d', 'ver_h', 'ver_m', 'ver_o']
    if allot_PRPR_VER in rem_punct_vers:
        re_punc = re.compile('[%s]' % re.escape(string.punctuation)) # remove punctuations
        tokens = [re_punc.sub('', w) for w in tokens] # remove punctuations

    rem_punc_split_vers = ['ver_b', 'ver_f', 'ver_g', 'ver_k', 'ver_n', 'ver_q']
    if allot_PRPR_VER in rem_punc_split_vers:
        re_punc = re.compile('[%s]' % re.escape(string.punctuation)) # remove punctuations
        tokens = [re_punc.sub(' ', w) for w in tokens] #
        temp=[]
        for each in tokens:
            temp += each.split()
        tokens = temp[:]


    ### (2) Preprocessing for filetering vocabulary size
    #########################################################
    # remove tokens not in vocab
    if IS_GENSIM == False:
        tokens = [w for w in tokens if w in VOCA]
        tokens = ' '.join(tokens)
    elif IS_GENSIM == True:
        tokens = [w for w in tokens if w in VOCA_gensim]

    return tokens



def save_txt(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

def data_preprocessing(MODEL, DATASET, PRPR_VER, vocab, IS_SAMPLE):
    global allot_PRPR_VER
    global VOCA
    allot_PRPR_VER = PRPR_VER
    VOCA = vocab

    # """ Load Dataset """
    train, test, len_info = data_load(
                                    DATASET,
                                    IS_SAMPLE)
    NUM_CLASSES = len_info[0]
    MAXnb_TOKENS_inD = len_info[1]
    MAXnb_SENTS_inD = len_info[2]
    MAXnb_TOKENS_inS = len_info[3]

    # """ Preparing Dataset """
    if MODEL == 'yang_rnn':
        # Make 3D shape (to distinguish sentence-level)
        ### Xtrain
        print('The number of training examples: ', len(train['X']))

        train_reviews = []
        train_temp_text_set = []
        temp = []
        for idx in range(train.X.shape[0]): # for train
            #train_temp_text_set.append(voca_clean(text))
            sentences = tokenize.sent_tokenize(train['X'][idx]) # tokenize sentences
            temp_sents = []
            for i,_ in enumerate(sentences):
                cleaned_sentence = voca_clean(sentences[i])
                temp_sents.append(cleaned_sentence)
                temp.append(cleaned_sentence)
            train_reviews.append(temp_sents)

        #print('\n')
        #save_txt(train_temp_text_set, 'train_temp_text_set.txt')
        #save_txt(temp, 'temp.txt')

        tokenizer = Tokenizer(lower=False, filters='')
        #tokenizer.fit_on_texts(train_temp_text_set) # to make indexed voca set using only train set.
        tokenizer.fit_on_texts(temp) # for python 2

        ### Keras Tokenizer Unicode Problem (python 2.7.12, keras 1.2.2)
        #(1) at keras.preprocessing.text... text_to_word_sequence function...
        #if lower: text = text.lower()
        #if type(text) == str:
        #    translate_table = {ord(c): ord(t) for c,t in zip(filters, split*len(filters)) }
        #else:
        #    translate_table = maketrans(filters, split * len(filters))
        #text = text.translate(translate_table)
        #seq = text.split(split)
        #return [i for i in seq if i]
        # (2) (we chose this!)
        ### Instead of the line 31:
        #text = text.translate(maketrans(filters, split*len(filters)))
        ### We need:
        #try :
        #    text = unicode(text, "utf-8")
        #except TypeError:
        #    pass
        #translate_table = {ord(c): ord(t) for c,t in  zip(filters, split*len(filters)) }
        #text = text.translate(translate_table)
        # (3)
        # train_temp_text_set = [s.encode('ascii') for s in train_temp_text_set]

        Xtrain = np.zeros((len(train['X']), MAXnb_SENTS_inD, MAXnb_TOKENS_inS), dtype='int32')
        for i, sentences in enumerate(train_reviews):
            for j, sent in enumerate(sentences):
                if j< MAXnb_SENTS_inD:
                    wordTokens = text_to_word_sequence(sent,lower=False, filters='') #### split!!!!!!!!!!1
                    k=0
                    #print(wordTokens)
                    for _, word in enumerate(wordTokens):
                        if k<MAXnb_TOKENS_inS:
                            Xtrain[i,j,k] = tokenizer.word_index[word]
                            k=k+1
        #print('\n')
        #print(Xtrain[:1])
        #print(Xtrain.shape)

        ### X_test
        test_reviews = []
        test_temp_text_set = []

        for idx in range(test.X.shape[0]): # for train
            sentences = tokenize.sent_tokenize(test['X'][idx]) # sentence tokenization
            temp_sents = []
            for i,_ in enumerate(sentences):
                temp_sents.append(voca_clean(sentences[i])) # preprecessing the results from sentence-tokenization
            test_reviews.append(temp_sents)

        X_test = np.zeros((len(test['X']), MAXnb_SENTS_inD, MAXnb_TOKENS_inS), dtype='int32')
        for i, sentences in enumerate(test_reviews):
            for j, sent in enumerate(sentences):
                if j< MAXnb_SENTS_inD:
                    wordTokens = text_to_word_sequence(sent,lower=False, filters='')
                    k=0
                    for _, word in enumerate(wordTokens):
                        if k<MAXnb_TOKENS_inS:
                            X_test[i,j,k] = tokenizer.word_index[word]
                            k=k+1
        ### ytrain & ytest
        ytrain = pd.get_dummies(np.asarray(train['y']))
        y_test = pd.get_dummies(np.asarray(test['y']))

        #print('\n')
        #print(y_test.shape)
        #print(y_test[:1])
        #print(_test.values[:1])

        
        np.random.seed(7)
        ### random shuffling data set
        indices = np.arange(Xtrain.shape[0])
        np.random.shuffle(indices)
        Xtrain = Xtrain[indices]
        ytrain = ytrain.values[indices]

        indices = np.arange(X_test.shape[0])
        np.random.shuffle(indices)
        X_test = X_test[indices]
        y_test = y_test.values[indices]

    else:
        ### Preprocessing Dataset
        ##########################
        train['X'] = train.X.apply(voca_clean)
        test['X'] = test.X.apply(voca_clean)

        ### Vectorize Dataset
        ##########################
        ## Define tokenizer
        tokenizer = create_tokenizer(train['X'])
        print('Found %s unique tokens.' % len(tokenizer.word_index))
        vocab_size = len(tokenizer.word_index) + 1 # +1 because of UNK token
        print('Vocabulary size: %d' % vocab_size)

        ## Encode data
        # through encode_docs function, we convert sequence words into sequence voca index
        Xtrain = encode_docs(tokenizer, MAXnb_TOKENS_inD, train['X'])
        ytrain = pd.get_dummies(train['y'])
        ytrain = ytrain.values
        print('Training Data Shape: ', Xtrain.shape, ytrain.shape)
        X_test = encode_docs(tokenizer, MAXnb_TOKENS_inD, test['X'])
        y_test = pd.get_dummies(test['y'])
        y_test = y_test.values
        #print(X_test.shape, y_test.shape)

        # random shuffle
        Xtrain, ytrain = shuffle(Xtrain, ytrain, random_state=7)

    return Xtrain, ytrain, X_test, y_test, tokenizer, len_info


#########################################
""" LOAD PRETRAINED WORD EMBEDDING """
#########################################

def load_pretrained_embedding(tokenizer_word_idx, pretrained_name, DATSET, PRPR_VER, EMBEDDING_DIM, IS_SAMPLE, vocab):
    global VOCA_gensim
    global IS_GENSIM
    VOCA_gensim = vocab
    # tokenizer_word_index: vocabulary of train set (word list to be converted into embedding)
    # pretrained_name: what kinds of embedding?

    cur_folder = os.getcwd().split(os.sep)[-1] # save current-path
    os.chdir('..') # move to parent-folder
    ABSOLUTE_PATH = os.getcwd()
    os.chdir('./'+str(cur_folder)) # return-to-original-path
    FULL_PATH = os.path.join(ABSOLUTE_PATH, 'embedding', "")

    emb_dim = EMBEDDING_DIM
    vocab_size = len(tokenizer_word_idx) + 1 # +1 becuase of UNK token

    #
    # glove_6b_300d
    if pretrained_name == 'glove_6b_300d':
        pretrained_f_name = 'glove.6B.300d.txt'
        FULL_PATH = os.path.join(FULL_PATH, 'glove.6B', "")
        INPUT_PATH = FULL_PATH + pretrained_f_name
        GENSIM_PATH = FULL_PATH + pretrained_f_name + '.word2vec'
        PICKLE_PATH = FULL_PATH+pretrained_name+'.'+DATSET+'.'+PRPR_VER+'.'+str(IS_SAMPLE)+'.pickle'

        # Is there pre-saved pickle?
        if os.path.isfile(PICKLE_PATH):
            emb_matrix = pickle_load(PICKLE_PATH)
            #print('Load pickle file!')
            assert(vocab_size==len(emb_matrix))
            return emb_matrix
        # if pickle is empty
        else:
            if os.path.isfile(GENSIM_PATH) == False: # if gensim is empty
                glove2word2vec(INPUT_PATH, GENSIM_PATH) # convert to gensim format
            else: # if gensim exist
                pass
            # Load gensim model
            model = KeyedVectors.load_word2vec_format(GENSIM_PATH, binary=False)

    #
    # glove_42b_300d
    elif pretrained_name == 'glove_42b_300d':
        pretrained_f_name = 'glove.42B.300d.txt'
        FULL_PATH = os.path.join(FULL_PATH, 'glove.42B', "")
        INPUT_PATH = FULL_PATH + pretrained_f_name
        GENSIM_PATH = FULL_PATH + pretrained_f_name + '.word2vec'
        PICKLE_PATH = FULL_PATH+pretrained_name+'.'+DATSET+'.'+PRPR_VER+'.'+str(IS_SAMPLE)+'.pickle'

        # Is there pre-saved pickle?
        if os.path.isfile(PICKLE_PATH):
            emb_matrix = pickle_load(PICKLE_PATH)
            #print('Load pickle file!')
            assert(vocab_size==len(emb_matrix))
            return emb_matrix
        # if pickle is empty
        else:
            if os.path.isfile(GENSIM_PATH) == False: # if gensim is empty
                glove2word2vec(INPUT_PATH, GENSIM_PATH) # convert to gensim format
            else: # if gensim exist
                pass
            # Load gensim model
            model = KeyedVectors.load_word2vec_format(GENSIM_PATH, binary=False)

    #
    # glove_840b_300d
    elif pretrained_name == 'glove_840b_300d':
        pretrained_f_name = 'glove.840B.300d.txt'
        FULL_PATH = os.path.join(FULL_PATH, 'glove.840B', "")
        INPUT_PATH = FULL_PATH + pretrained_f_name
        GENSIM_PATH = FULL_PATH + pretrained_f_name + '.word2vec'
        PICKLE_PATH = FULL_PATH+pretrained_name+'.'+DATSET+'.'+PRPR_VER+'.'+str(IS_SAMPLE)+'.pickle'

        # Is there pre-saved pickle?
        if os.path.isfile(PICKLE_PATH):
            emb_matrix = pickle_load(PICKLE_PATH)
            #print('Load pickle file!')
            assert(vocab_size==len(emb_matrix))
            return emb_matrix
        # if pickle is empty
        else:
            if os.path.isfile(GENSIM_PATH) == False: # if gensim is empty
                glove2word2vec(INPUT_PATH, GENSIM_PATH) # convert to gensim format
            else: # if gensim exist
                pass
            # Load gensim model
            model = KeyedVectors.load_word2vec_format(GENSIM_PATH, binary=False)
			
	#
    # skip_word2vec_300d
    elif pretrained_name == 'skip_word2vec_300d':
        pretrained_f_name = 'GoogleNews-vectors-negative300.bin' # this is binary file!
        FULL_PATH = os.path.join(FULL_PATH, 'skip-gram', "")
        INPUT_PATH = FULL_PATH + pretrained_f_name
        #GENSIM_PATH # not used in skip-gram
        PICKLE_PATH = FULL_PATH+pretrained_name+'.'+DATSET+'.'+PRPR_VER+'.'+str(IS_SAMPLE)+'.pickle'

        if os.path.isfile(PICKLE_PATH):
            emb_matrix = pickle_load(PICKLE_PATH)
            assert(vocab_size==len(emb_matrix))
            return emb_matrix
        else: # if pickle is empty
            # we don't need gensim model for skip-gram
            model = KeyedVectors.load_word2vec_format(INPUT_PATH, binary=True) # binary=True!!!
    #
    # fasttext_300d
    elif pretrained_name == 'fasttext_300d':
        #pretrained_f_name = 'wiki-news-300d-1M-subword.vec'
        pretrained_f_name = 'wiki.en.vec'
        FULL_PATH = os.path.join(FULL_PATH, 'fastText', "")
        INPUT_PATH = FULL_PATH + pretrained_f_name
        #GENSIM_PATH: not used in fastText
        PICKLE_PATH = FULL_PATH+pretrained_name+'.'+DATSET+'.'+PRPR_VER+'.'+str(IS_SAMPLE)+'.pickle'

        if os.path.isfile(PICKLE_PATH):
            emb_matrix = pickle_load(PICKLE_PATH)
            assert(vocab_size==len(emb_matrix))
            return emb_matrix
        else: # if pickle is empty
            # we don't need gensim model for fastText
            model = KeyedVectors.load_word2vec_format(INPUT_PATH, binary=False)
    #
    elif pretrained_name == 'rand_300d': # this is not the pretrained embedding.
        return None

    elif pretrained_name == 'average_300d':
        return None

    elif pretrained_name == 'gensim_skip_300d': # this is not the pretrained embedding.
        IS_GENSIM = True
        pretrained_name = 'gensim_skip_300d'
        INPUT_PATH = os.path.join(FULL_PATH, 'gensim-skip', "") # 'gensim_skip' folder
        PICKLE_PATH =INPUT_PATH+pretrained_name+'.'+DATSET+'.'+PRPR_VER+'.'+str(IS_SAMPLE)+'.pickle'

        if os.path.isfile(PICKLE_PATH):
            emb_matrix = pickle_load(PICKLE_PATH)
            assert(vocab_size==len(emb_matrix))
            return emb_matrix
        else: # if pickle is empty
            # train skip-gram model with the specific dataset.

            train, _, _ = data_load(DATSET,False) # load only train set.
            train['X'] = train.X.apply(voca_clean)
            sentences = train['X'].tolist()
            model = Word2Vec(sentences, size=300, window=10, min_count=0, sg=1) # sg1: skip-gram

    elif pretrained_name == 'gensim_fast_300d': # this is not the pretrained embedding.
        IS_GENSIM = True
        pretrained_name = 'gensim_fast_300d'
        INPUT_PATH = os.path.join(FULL_PATH, 'gensim-fast', "") # 'gensim_skip' folder
        PICKLE_PATH =INPUT_PATH+pretrained_name+'.'+DATSET+'.'+PRPR_VER+'.'+str(IS_SAMPLE)+'.pickle'

        if os.path.isfile(PICKLE_PATH):
            emb_matrix = pickle_load(PICKLE_PATH)
            assert(vocab_size==len(emb_matrix))
            return emb_matrix
        else: # if pickle is empty
            # train skip-gram model with the specific dataset.

            train, _, _ = data_load(DATSET,False) # load only train set.
            train['X'] = train.X.apply(voca_clean)
            sentences = train['X'].tolist()
            model = FastText(sentences, size=300, window=10, min_count=0, sg=1) # sg1: skip-gram

    else:
        raise SystemExit('Specified word embedding can not be found! (Check the folder)')

    ##############################################################################
    cnt = 0
    # create a weight matrix for words in training docs
    #embedding_matrix = np.random.random((vocab_size, emb_dim))
    #embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, emb_dim))
    embedding_matrix = np.zeros((vocab_size, emb_dim))
    for word, i in tokenizer_word_idx.items(): # idx 1 ~ len(tokenizer_word_idx)
        try:
            # insert word embedding
            embedding_matrix[i] = model[word] # model[word] exist?
        except:
            cnt += 1
            embedding_matrix[i] = np.random.uniform(-0.25, 0.25, 300)
            continue # if there is no word in word-embedding-model, just pass

    print('\n')
    print('--> ', pretrained_name)
    print('# total vocab (except for unknown token): ', len(tokenizer_word_idx))
    print('# not in pretrained:', cnt)

    pickle_save(embedding_matrix, PICKLE_PATH)
    return embedding_matrix



###############
""" etc. """
###############

# text encoding
def create_tokenizer(lines):
    tokenizer = Tokenizer(lower=False, filters='')
    tokenizer.fit_on_texts(lines) # input: list of text
    return tokenizer

def encode_docs(tokenizer, max_length, docs):
    encoded = tokenizer.texts_to_sequences(docs) # input: list of text
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded

# related to model
def argmax(arr): return np.argmax(arr)

# load and store
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read() # read all
    file.close()
    return text

import sys
def pickle_save(data, name_path):
    filehandler = open(name_path,"wb")
    if (sys.version_info > (3, 0)): # python 3
        pickle.dump(data, filehandler)
    else: # python 2
        pickle.dump(data, filehandler, protocol=2)
    filehandler.close()

def pickle_load(name_path):
    filehandler = open(name_path, "rb")
    return pickle.load(filehandler)
