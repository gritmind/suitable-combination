# Training dataset에 대한 (주로 통계) 분석을 실시 (결과 저장)
# Training dataset에 전처리를 실시한 후 vocabulary를 만듦
# vocab를 무조건 미리 만들어야 한다. 왜냐하면 vocab는 training data에 기반해서 만들어지기 때문이다. 
# training data에 없는 단어가 test data에 있는 경우 (전처리를 거쳐서 살아남을지라도) vocab에 의해 삭제된다.

import pandas as pd
import re
import os
import sys
import string
import errno
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import argparse

ps = PorterStemmer()
vocab = Counter()
global MIN_FREQ


parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
parser.add_argument('--agnews', action='store_true')
parser.add_argument('--yelpp', action='store_true')
parser.add_argument('--verone', action='store_true')
args = parser.parse_args()

# argparse는 append하게 arg들을 추가할 수 있다
# 그러나 한 가지 종류의 arg에 대해서는 정의할 떄는 여러 개로 할 수 있지만 불러올 떄는 한 가지만 불러오기 위해서 다음과 같이 if, elif문으로 구성하였다 
# 만약 dataset에 관한 arg가 여러 개 입력했을 경우, elif문에서 가장 상위에 있는 arg.dataset이 우선적으로 할당된다

# (1) dataset type
if args.agnews == True:
    DATASET = 'agnews'
elif args.yelpp == True:
    DATASET = 'yelpp'
else:
    print("[arg error!] please add arg: python 1_build-vocab --(dataset-name)")
    exit()

# (2) preprocessing version
if args.verone == True:
    PRPR_VER = 'verone'    
else:
    print("[arg error!] please add arg: python 1_build-vocab --(version)")
    exit()    
      
### Path setting
ABSOLUTE_PATH = os.getcwd()
FULL_PATH = ABSOLUTE_PATH+'\\'+'dataset-description'+'\\'+DATASET + '\\'

# Full path에 맞게 폴더를 모두 생성. (recursive하게 모두 생성됨.)
if not os.path.exists(os.path.dirname(FULL_PATH)):
    try:
        os.makedirs(os.path.dirname(FULL_PATH))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

#print(FULL_PATH)
# 여기까지 저장할 폴더, 이름을 다 준비해놓는 코드.
# FULL_PATH에 Model관련 정보파일들, Description 파일, (2)Result 파일, .png 파일들이 저장된다.
############################################################################################ 
############################################################################################
############################################################################################
## START stdout for Description file
orig_stdout = sys.stdout
f = open(FULL_PATH+'description.txt', 'w')
sys.stdout = f


#################################
""" DATA EXPLORATION """
#################################
def data_statistic_analysis(train):
    print('\n\n < Data Statistic Anaylsis >\n')
    ### dataset exploration

    ## corpus 기준
    # class 총 개수
    # document 총 개수
    # vocubulary size
    # (전처리 후) vocabulary size

    ## doc 기준
    # sent 평균 개수
    # sent 최대 개수
    # sent 최소 개수
    # word 평균 개수
    # word 최대 개수
    # word 최소 개수

    # 만약 데이터가 train/test로 제공되어 진다면, 전체 셋을 대상으로 하고 train 대상, test 대상 따로도 해보자.
    pass

##########################################
""" Text Cleaning to build Vocabulary """
##########################################
def text_cleaning_for_voca(doc):
    global MIN_FREQ
    tokens = doc.split()  
    
    if PRPR_VER == 'verone': 
        """ TEXT_PREPROCESSING_VER (=PRPR_VER)"""
        """ ver00 """
        ### 
        #MIN_FREQ = 3 # agnews: 32405
        #MIN_FREQ = 5 # agnews: 23615
        MIN_FREQ = 3
        ### (1) 모양만 바뀌는 전처리
        #######################    
            
        # lower
        tokens = [word.lower() for word in tokens]
        # stemming
        tokens = [ps.stem(word) for word in tokens]    
        # remove punctuations
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        tokens = [re_punc.sub('', w) for w in tokens]
        
        ### (2) 삭제를 위한 전처리
        #######################    
        # remove non-alphabetic tokens
        tokens = [word for word in tokens if word.isalpha()]
        # remove stop-words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if not word in stop_words]
        # remove non-freq words (주의: document내에서 한정)
        # AG'news는 document 길이가 짧기 때문에 실시하지 않음.
        #tokens = [word for word in tokens if len(word) > 1]
        

    if PRPR_VER == 'vertwo':
        pass
    
    return tokens

    
    
    
################################################################################
""" MAIN  """
################################################################################
    
# train dataset만 로드하자.    
if args.agnews == True:
    train = pd.read_csv('../dataset/ag_news_csv/train.csv', header=None)
    data_statistic_analysis(train) # 데이터 통계 분석 (결과는 printf로 출력(file에 저장))
    
    train.columns = ['label', 'title', 'description'] # column rename
    # preprocessing using 'text_cleaning_for_voca' function
    train['title'] = train.title.apply(text_cleaning_for_voca)
    train['description'] = train.description.apply(text_cleaning_for_voca)
    # vocab variable
    for idx, row in train.iterrows():
        vocab.update(row['title'])
        vocab.update(row['description'])
    print('# of unique words in training dataset: ', len(vocab))
    #print(vocab.most_common(10))
    tokens = [k for k,c in vocab.items() if c >= MIN_FREQ]
    print('Min_freq_threshold: ', MIN_FREQ)
    print('(After pruning) vocabulary size: ', len(tokens))
    #vocab = set(tokens)

elif args.yelpp == True:
    pass

## END stdout for Description file 
sys.stdout = orig_stdout
f.close()

print('\n')
print('>> [%s] data description is saved!' % DATASET)  
  
def save_vocab(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
    
### save
VOCAB_FILE_NAME = 'vocab_'+PRPR_VER+'.txt'
VOCAB_PATH = FULL_PATH + VOCAB_FILE_NAME
if not os.path.isfile(VOCAB_PATH):
    save_vocab(tokens, VOCAB_PATH)
print('>> [%s] vocab is saved!' % DATASET) 

# [참고] Load vocabulary
#vocab_filename = 'voca.txt'
#vocab = load_doc(vocab_filename)
#vocab = set(vocab.split())































