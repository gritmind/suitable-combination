from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors 
from gensim.models.wrappers import FastText
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
import pickle
import string
import os
import re
ps = PorterStemmer()
global allot_PRPR_VER
global VOCA



######################################################
""" LOAD & PREPROCESSING & VECTORIZING DATASET """
######################################################

def voca_clean(doc): # 여기서는 주로 모양만 바뀌는 전처리를 하고, vocab 필터링을 한다.
    global allot_PRPR_VER
    global VOCA
    tokens = doc.split()

    if allot_PRPR_VER == 'verone':
        ### (1) 모양만 바뀌는 전처리
        #######################
        # lower
        tokens = [word.lower() for word in tokens]
        # stemming
        tokens = [ps.stem(word) for word in tokens]    
        # remove punctuations
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        tokens = [re_punc.sub('', w) for w in tokens]
        
        ### (2) vocab를 통한 필터링
        #######################    
        # remove tokens not in vocab
        tokens = [w for w in tokens if w in VOCA]
        tokens = ' '.join(tokens)
    
    if allot_PRPR_VER == 'vertwo':
        pass
    
    return tokens

def data_preprocessing(DATA_TYPE, PRPR_VER, vocab, train_path, test_path, IS_SAMPLE):
    global allot_PRPR_VER
    global VOCA
    allot_PRPR_VER = PRPR_VER
    VOCA = vocab
    
    """ Load Dataset """
    #
    # agnews
    if DATA_TYPE == 'agnews':
        if IS_SAMPLE == True:
            train = pd.read_csv(train_path, header=None)
        elif IS_SAMPLE == False:
            train = pd.read_csv(train_path, header=None)
        test = pd.read_csv(test_path, header=None)

        ## column rename
        train.columns = ['y', 'title', 'description']
        test.columns = ['y', 'title', 'description']
        
        ## merge and drop
        # merge
        train['X'] = train['title'] + ' ' + train['description']
        test['X'] = test['title'] + ' ' + test['description']
        # drop
        train = train.drop(['title'], axis=1)
        train = train.drop(['description'], axis=1)
        test = test.drop(['title'], axis=1)
        test = test.drop(['description'], axis=1) 
        
        # 하나의 document에 최대 word 개수 설정
        max_length = max([len(s.split()) for s in train['X']])
        # 일반적으로 위와 같이 학습데이터 중에서 가장 긴 사이즈를 max_length로 잡으면 되지만
        # 모델에 따라 최소 max_length를 요구하는 경우가 있다
        # 따라서 우리는 여기서 max_length를 임의로 잡는다
        max_length = 300 # 데이터셋에 따라 달라질 것이다 (Agnews: 300)
        print('Maximum length: %d' % max_length)    
    #
    # agnews
    if DATA_TYPE == 'agnews':
        pass    
    
    
    # 데이터셋 종류에 상관없이 공통적으로 처리되는 부분.
    ################################################################################
    """ Preprocessing Dataset """
    train['X'] = train.X.apply(voca_clean)
    test['X'] = test.X.apply(voca_clean)

    """ Vectorize Dataset """
    ## Define tokenizer
    tokenizer = create_tokenizer(train['X'])
    print('Found %s unique tokens.' % len(tokenizer.word_index))
    vocab_size = len(tokenizer.word_index) + 1 # UNK token 때문에 +1을 해준다
    print('Vocabulary size: %d' % vocab_size)   
    
    ## Encode data    
    # encode_docs 함수를 통해 sequence words를 sequence voca index로 변환해준다
    Xtrain = encode_docs(tokenizer, max_length, train['X'])
    ytrain = pd.get_dummies(train['y'])
    print('Training Data Shape: ', Xtrain.shape, ytrain.shape)
    X_test = encode_docs(tokenizer, max_length, test['X'])
    y_test = pd.get_dummies(test['y'])
    #print(X_test.shape, y_test.shape)

    # random shuffle
    Xtrain, ytrain = shuffle(Xtrain, ytrain, random_state=7)    

    return Xtrain, ytrain, X_test, y_test, tokenizer      



            
            
            
            


#########################################
""" LOAD PRETRAINED WORD EMBEDDING """
#########################################

def load_pretrained_embedding(tokenizer_word_idx, pretrained_name, DATSET, PRPR_VER, EMBEDDING_DIM):
    
    # tokenizer_word_index: 현재 데이터셋에서의 vocabulary (or 우리가 embedding으로 변환해야될 단어 리스트)
    # pretrained_name: embedding 종류 명시
    
    cur_folder = os.getcwd().split(os.sep)[-1] # 현재 path로 다시 원상복구시키기 위해 현재 폴더이름 저장
    os.chdir('..') # 현재 디렉토리에서 상위 폴더로 이동하기 위해
    ABSOLUTE_PATH = os.getcwd()
    os.chdir('./'+str(cur_folder)) # 다시 원상복구
    FULL_PATH = ABSOLUTE_PATH+'\\'+'embedding'+'\\'    
    
    emb_dim = EMBEDDING_DIM
    vocab_size = len(tokenizer_word_idx) + 1 # # UNK token 때문에 +1을 해준다

    #
    # glove_6b_300d
    if pretrained_name == 'glove_6b_300d':
        pretrained_f_name = 'glove.6B.300d.txt'
        FULL_PATH += 'glove.6B\\'
        INPUT_PATH = FULL_PATH + pretrained_f_name
        GENSIM_PATH = FULL_PATH + pretrained_f_name + '.word2vec'
        PICKLE_PATH = FULL_PATH+pretrained_name+'.'+DATSET+'.'+PRPR_VER+'.pickle'
        
                
        # 미리 pickle로 저장되어 있는지 확인.
        # [불러올 때 조심] sample기반으로 embedding_mat를 생성한 것을 실제실험에 사용하면 안됨!
        
        if os.path.isfile(PICKLE_PATH):
            emb_matrix = pickle_load(PICKLE_PATH)
            #print('Load pickle file!')
            assert(vocab_size==len(emb_matrix)) 
            return emb_matrix
        else: # pickle파일이 없다면.
            # gensim format으로 바꾸자.
            if os.path.isfile(GENSIM_PATH) == False:
                glove2word2vec(INPUT_PATH, GENSIM_PATH) # gensim format으로 변환
            else: # gensim format이 있다면..
                pass
            # gensim model 불러오기
            model = KeyedVectors.load_word2vec_format(GENSIM_PATH, binary=False) 
    #
    # skip_word2vec_300d
    elif pretrained_name == 'skip_word2vec_300d':
        pretrained_f_name = 'GoogleNews-vectors-negative300.bin' # binary file임을 인지하자!
        FULL_PATH += 'skip-gram\\'
        INPUT_PATH = FULL_PATH + pretrained_f_name    
        #GENSIM_PATH: skip-gram에서는 정의되지 않는다.
        PICKLE_PATH = FULL_PATH+pretrained_name+'.'+DATSET+'.'+PRPR_VER+'.pickle'

        if os.path.isfile(PICKLE_PATH):
            emb_matrix = pickle_load(PICKLE_PATH)
            assert(vocab_size==len(emb_matrix))
            return emb_matrix
        else: # pickle파일이 없다면.
            # 따로 gensim format으로 바꿔줄 필요가 없다 (input file자체가 gensim과 혼용된다)
            model = KeyedVectors.load_word2vec_format(INPUT_PATH, binary=True) # 주의! binary=True
    #
    # fasttext_300d        
    elif pretrained_name == 'fasttext_300d':
        pretrained_f_name = 'wiki-news-300d-1M.vec'
        FULL_PATH += 'fastText\\'
        INPUT_PATH = FULL_PATH + pretrained_f_name         
        #GENSIM_PATH: fastText에서는 정의되지 않는다.
        PICKLE_PATH = FULL_PATH+pretrained_name+'.'+DATSET+'.'+PRPR_VER+'.pickle'     
        
        if os.path.isfile(PICKLE_PATH):
            emb_matrix = pickle_load(PICKLE_PATH)
            assert(vocab_size==len(emb_matrix))
            return emb_matrix
        else: # pickle파일이 없다면.
            # 따로 gensim format으로 바꿔줄 필요가 없다 (input file자체가 gensim과 혼용된다)
            model = KeyedVectors.load_word2vec_format(INPUT_PATH)
    #
    else:
        raise SystemExit('Specified word embedding can not be found! (Check the folder)') 
    
    ## ----- 여기까지 embedding model 정의 (변수 model에 할당)
    ##############################################################################
    
    # create a weight matrix for words in training docs 
    embedding_matrix = np.random.random((vocab_size, emb_dim))
    for word, i in tokenizer_word_idx.items():
        try: embedding_vector = model[word]
        except: continue # if there is no word in word-embedding-model, just pass
        # insert word embedding
        embedding_matrix[i] = embedding_vector        
    
    # pickle로 저장
    pickle_save(embedding_matrix, PICKLE_PATH)
    return embedding_matrix
        

###############
""" etc. """
###############

# text encoding

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines) # 입력: list of text
    return tokenizer

def encode_docs(tokenizer, max_length, docs):
    encoded = tokenizer.texts_to_sequences(docs) # 입력: list of text
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded

    
# related to model

def argmax(arr):
    return np.argmax(arr)

   
# load and store

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read() # read all
    file.close()
    return text

def pickle_save(data, name_path):
    filehandler = open(name_path,"wb")
    pickle.dump(data, filehandler)
    filehandler.close()
    
def pickle_load(name_path):
    filehandler = open(name_path, "rb")
    return pickle.load(filehandler)





    
    
    
