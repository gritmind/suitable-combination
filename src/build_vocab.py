# -*- coding: utf-8 -*-
# Statistical Analysis for training dataset
# after preprocessing training data, we build vocabulary.
global DEBUG
DEBUG = False # if true, show cmd.

import pandas as pd
import re
import os
import sys
import string
import errno
from collections import Counter
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import tokenize
import argparse
global ps
global vocab
global GLOBALOFF
global RATIO
ps = PorterStemmer()
lm = WordNetLemmatizer()
vocab = Counter()


##########################################
""" TEXT CLEANING """
##########################################
def text_cleaning_for_voca(doc):
    # chose only one
    global MIN_FREQ
    global RATIO
    global GLOBALOFF
    global MOST_COMMON_NUM
    global PRPR_VER
    global ps
    allot_PRPR_VER = PRPR_VER
    tokens = doc.split()

    GLOBALOFF = False
	
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
  
    ###################################################################################
    ### (2)
    MIN_FREQ = 3
    #tokens = [word for word in tokens if word.isalpha()] # remove non-alphabetic tokens
    stop_words = set(stopwords.words('english')) # remove stop-words
    tokens = [word for word in tokens if not word in stop_words]
    # remove non-freq words (limit to document-level)
    #tokens = [word for word in tokens if len(word) > 1]

    return tokens


###########################################
""" DATA LOAD  """
###########################################

def data_load(DATASET, IS_SAMPLE):
    if DATASET == 'agnews':

        # data load
        if IS_SAMPLE == True:
            train = pd.read_csv('../dataset/ag_news_csv/sample.csv', header=None)
        elif IS_SAMPLE == False:
            train = pd.read_csv('../dataset/ag_news_csv/train.csv', header=None)
        test = pd.read_csv('../dataset/ag_news_csv/test.csv', header=None) # test set은 is only for statistical analysis
        #^^; since ag-news dataset does not have NA data, we do not need to have 'keep_default_na=False'

        ## column rename
        train.columns = ['y', 'title', 'description']
        test.columns = ['y', 'title', 'description']
        # merge (merge with '.' to recognize different sentence)
        train['X'] = train['title'] + '. ' + train['description']
        test['X'] = test['title'] + '. ' + test['description']

        #! Write down after examining data-description folder.
        NUM_CLASSES = 4
        MAXnb_TOKENS_inD = 300 # (Agnews: 300)
        MAXnb_SENTS_inD =  20
        MAXnb_TOKENS_inS = 100

    elif DATASET == 'yelpp':
        # data load
        if IS_SAMPLE == True:
            pass
        elif IS_SAMPLE == False:
            train = pd.read_csv('../dataset/yelp_review_polarity_csv/train.csv', header=None, keep_default_na=False)
        test = pd.read_csv('../dataset/yelp_review_polarity_csv/test.csv', header=None, keep_default_na=False)

        ## column rename
        train.columns = ['y', 'X']
        test.columns = ['y', 'X']

        NUM_CLASSES = 2
        MAXnb_TOKENS_inD = 1052 # 1052
        MAXnb_SENTS_inD =  141 # 141
        MAXnb_TOKENS_inS = 762

    elif DATASET == 'yahoo':
        # data load
        if IS_SAMPLE == True:
            pass
        elif IS_SAMPLE == False:
            train = pd.read_csv('../dataset/yahoo_answers_csv/train.csv', header=None, keep_default_na=False)
        test = pd.read_csv('../dataset/yahoo_answers_csv/test.csv', header=None, keep_default_na=False)

        ## column rename
        train.columns = ['y', 'question_title', 'question_content', 'best_answer']
        test.columns = ['y', 'question_title', 'question_content', 'best_answer']
        # merge (merge with '.' to recognize different sentence)
        train['X'] = train['question_title'] + '. ' + train['question_content'] + '. ' + train['best_answer']
        test['X'] = test['question_title'] + '. ' + test['question_content'] + '. ' + test['best_answer']

        NUM_CLASSES = 10
        MAXnb_TOKENS_inD = 1100 # 1052
        MAXnb_SENTS_inD =  150 # 141
        MAXnb_TOKENS_inS = None

    elif DATASET == 'dbpedia':
        # data load
        if IS_SAMPLE == True:
            pass
        elif IS_SAMPLE == False:
            train = pd.read_csv('../dataset/dbpedia_csv/train.csv', header=None)
        test = pd.read_csv('../dataset/dbpedia_csv/test.csv', header=None)

        ## column rename
        train.columns = ['y', 'title', 'abstract']
        test.columns = ['y', 'title', 'abstract']
        # merge (merge with '.' to recognize different sentence)
        train['X'] = train['title'] + '. ' + train['abstract']
        test['X'] = test['title'] + '. ' + test['abstract']

        NUM_CLASSES = 10
        MAXnb_TOKENS_inD = 1100 # 1052
        MAXnb_SENTS_inD =  150 # 141
        MAXnb_TOKENS_inS = None

        df.fillna('')

    return train, test, [NUM_CLASSES, MAXnb_TOKENS_inD, MAXnb_SENTS_inD, MAXnb_TOKENS_inS]

def save_vocab(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

    
#################################
""" DATA EXPLORATION """
#################################
def data_statistic_analysis(data, TRAIN, PREP):
    print('\n[ Data Statistic Anaylsis ]')
    if TRAIN == 'train':
        print('(1) [Train] Data')
    elif TRAIN == 'test':
        print('(1) [Test] Data')
    if PREP == 'before-pre':
        print('(2) Before Preprocessing')
    elif PREP == 'after-pre':
        print('(2) After Preprocessing')

    ### DATA EXPLORATION
    ### (always, data has data['X'] and data['y'])
    data['#_token'] = data.apply(lambda row: len(row['X'].split()), axis=1)
    data['#_sent'] = data.apply(lambda row: len(tokenize.sent_tokenize(row['X'])), axis=1)

    ## corpus criteria
    print('\n-> Corpus based')
    print('Sum of all tokens: ', data['#_token'].sum())
    print('Sum of all sentences: ', data['#_sent'].sum())

    ## document criteria
    print('\n-> Document based')
    print('  max===>',data['#_token'].max())
    print(data['#_token'].describe())
    print('\n')
    print('  max===>',data['#_sent'].max())
    print(data['#_sent'].describe())

    ## sentence criteria
    print('\n-> Sentence based')
    temp_text_set = []
    for idx in range(data.X.shape[0]):
        text = data['X'][idx]
        temp_text_set += tokenize.sent_tokenize(text)
    temp_df = pd.DataFrame({'sent':temp_text_set})
    temp_df['#_word_in_a_sent'] = temp_df.apply(lambda row: len(row['sent'].split()), axis=1)
    print('  max===>', temp_df['#_word_in_a_sent'].max())
    print(temp_df['#_word_in_a_sent'].describe())

    pass



###########################################
""" MAIN  """
###########################################
def main():
    global MOST_COMMON_NUM
    global MIN_FREQ
    global PRPR_VER
    global ps
    global DEBUG
    global RATIO
    global GLOBALOFF

    parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
    # (0) is sample or not
    #parser.add_argument('--sample', action='store_true')
    #parser.add_argument('--data', action='store_true')
    # (1) dataset type
    parser.add_argument('--agnews', action='store_true')
    parser.add_argument('--yelpp', action='store_true')
    parser.add_argument('--yahoo', action='store_true')
    parser.add_argument('--dbpedia', action='store_true')
    # (2) preprocessing version
    parser.add_argument('--ver_a', action='store_true')
    parser.add_argument('--ver_b', action='store_true')
    parser.add_argument('--ver_c', action='store_true')
    parser.add_argument('--ver_d', action='store_true')
    parser.add_argument('--ver_e', action='store_true')
    parser.add_argument('--ver_f', action='store_true')
    parser.add_argument('--ver_g', action='store_true')
    parser.add_argument('--ver_h', action='store_true')
    parser.add_argument('--ver_i', action='store_true')
    parser.add_argument('--ver_j', action='store_true')
    parser.add_argument('--ver_k', action='store_true')
    parser.add_argument('--ver_l', action='store_true')
    parser.add_argument('--ver_m', action='store_true')
    parser.add_argument('--ver_n', action='store_true')
    parser.add_argument('--ver_o', action='store_true')
    parser.add_argument('--ver_p', action='store_true')
    parser.add_argument('--ver_q', action='store_true')
    parser.add_argument('--ver_r', action='store_true')
    # (0) is sample?
    #if parser.parse_args().sample == True:
    #    IS_SAMPLE = True
    #elif parser.parse_args().data == True:
    #    IS_SAMPLE = False

    # (1) dataset type
    if parser.parse_args().agnews == True: DATASET = 'agnews'
    elif parser.parse_args().yelpp == True: DATASET = 'yelpp'
    elif parser.parse_args().yahoo == True: DATASET = 'yahoo'
    elif parser.parse_args().dbpedia == True: DATASET = 'dbpedia'
    else:
        print("[arg error!] please add arg: python 1_build-vocab --(dataset-name)")
        exit()

    # (2) preprocessing version
    if parser.parse_args().ver_a == True: PRPR_VER = 'ver_a'
    elif parser.parse_args().ver_b == True: PRPR_VER = 'ver_b'
    elif parser.parse_args().ver_c == True: PRPR_VER = 'ver_c'
    elif parser.parse_args().ver_d == True: PRPR_VER = 'ver_d'
    elif parser.parse_args().ver_e == True: PRPR_VER = 'ver_e'
    elif parser.parse_args().ver_f == True: PRPR_VER = 'ver_f'
    elif parser.parse_args().ver_g == True: PRPR_VER = 'ver_g'
    elif parser.parse_args().ver_h == True: PRPR_VER = 'ver_h'
    elif parser.parse_args().ver_i == True: PRPR_VER = 'ver_i'
    elif parser.parse_args().ver_j == True: PRPR_VER = 'ver_j'
    elif parser.parse_args().ver_k == True: PRPR_VER = 'ver_k'
    elif parser.parse_args().ver_l == True: PRPR_VER = 'ver_l'
    elif parser.parse_args().ver_m == True: PRPR_VER = 'ver_m'
    elif parser.parse_args().ver_n == True: PRPR_VER = 'ver_n'
    elif parser.parse_args().ver_o == True: PRPR_VER = 'ver_o'
    elif parser.parse_args().ver_p == True: PRPR_VER = 'ver_p'
    elif parser.parse_args().ver_q == True: PRPR_VER = 'ver_q'
    elif parser.parse_args().ver_r == True: PRPR_VER = 'ver_r'        
    else:
        print("[arg error!] please add arg: python 1_build-vocab --(version)")
        exit()

    ### Path setting
    ABSOLUTE_PATH = os.getcwd()
    FULL_PATH = os.path.join(ABSOLUTE_PATH, 'dataset-description', DATASET, "")

    # Folder Generation based on Full path (recursively)
    if not os.path.exists(os.path.dirname(FULL_PATH)):
        try:
            os.makedirs(os.path.dirname(FULL_PATH))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    #print(FULL_PATH)

    ############################################################################################
    ## START stdout for Description file
    if DEBUG == False:
        orig_stdout = sys.stdout
        f = open(FULL_PATH+'description___'+PRPR_VER+'.txt', 'w')
        sys.stdout = f

    ### Data load
    train, test, _ = data_load(DATASET, False) # False when making vocabulary

    ###
    ### common process (regardness of dataset)
    # vocab = list((word, freq))).

    # data statistics (before preprocessing)
    data_statistic_analysis(train, 'train' , 'before-pre')
    data_statistic_analysis(test , 'test', 'before-pre')

    """ Text Preprocessing """
    # preprocessing using 'text_cleaning_for_voca' function
    train['X'] = train.X.apply(text_cleaning_for_voca)
    test['X'] = test.X.apply(text_cleaning_for_voca)

    # data statistics (after preprocessing)
    data_statistic_analysis(train, 'train' , 'after-pre')
    data_statistic_analysis(test , 'test', 'after-pre')

    # vocab variable
    for idx, row in train.iterrows(): # using only train set when building vocabulary
        vocab.update(row['X'])

    # always execute CUTOFF
    #MIN_FREQ = 3 # fixed number.
    print('\n\nMin_freq (cutoff) threshold: ', MIN_FREQ)
    tokens = [k for k,c in vocab.items() if c >= MIN_FREQ]

    if GLOBALOFF == True:
        print('original vocab size: ', len(tokens))
        MOST_COMMON_NUM = int(round(len(tokens) * RATIO))
        print('filtered vocab size with ratio ',RATIO, ': ', MOST_COMMON_NUM)
        tokens = [tuple_[0] for tuple_ in vocab.most_common(MOST_COMMON_NUM)]

    print('(FINAL) vocab size: ', len(tokens))
    #vocab = set(tokens)

    ## END stdout for Description file
    if DEBUG == False:
        sys.stdout = orig_stdout
        f.close()

    print('>> [ %s ] data description is saved!' % DATASET)

    ### save
    VOCAB_FILE_NAME = 'vocab_'+PRPR_VER+'.txt'
    VOCAB_PATH = FULL_PATH + VOCAB_FILE_NAME
    if not os.path.isfile(VOCAB_PATH):
        save_vocab(tokens, VOCAB_PATH)
    print('>> [',DATASET,'] & [', PRPR_VER,'] vocab is saved!')

    # [참고] Load vocabulary
    #vocab_filename = 'voca.txt'
    #vocab = load_doc(vocab_filename)
    #vocab = set(vocab.split())

if __name__ == "__main__":
    main()
