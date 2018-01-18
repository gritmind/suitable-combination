
### EXPLICIT PARAMETER
##########################################
EMBEDDING_DIM = 300    
BATCH_SIZE = 128 # 3가지 모델 모두 통일하자.    
NB_EPOCHS = 1
NB_REAL_EX = 2 # 실제 실험 횟수 
MODEL = 'kim_cnn' 

USE_VAL_SET = False # false이면 val-set을 따로 분할해서 사용하지 않고 NB_FOLDS 변수는 의미 없어짐. 
VALIDATION_SPLIT = 0.1
NB_FOLDS = 10      

from data_handler import *

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from time import gmtime, strftime

import numpy as np
import string
import sys
import os
import errno
import argparse

ps = PorterStemmer()
seed = 7
np.random.seed(seed)
parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")

####################################
""" PARSING STETTING """
####################################

# dataset
parser.add_argument('--agnews', action='store_true')
parser.add_argument('--yelpp', action='store_true')
# preprocessing version
parser.add_argument('--verone', action='store_true')
parser.add_argument('--vertwo', action='store_true')
# word embedding
parser.add_argument('--skip_word2vec_300d', action='store_true')
parser.add_argument('--glove_6b_300d', action='store_true')
parser.add_argument('--fasttext_300d', action='store_true')
# word-embedding trainable check
parser.add_argument('--trainable', action='store_true')
parser.add_argument('--nontrainable', action='store_true')


# dataset argument parsing
if parser.parse_args().agnews == True:
    DATASET = 'agnews'
    NUM_CLASSES = 4
    PATH_TRAIN = '../dataset/ag_news_csv/train.csv'
    PATH_SAMPLE = '../dataset/ag_news_csv/sample.csv'
    PATH_TEST = '../dataset/ag_news_csv/test.csv'
elif parser.parse_args().yelpp == True:
    DATASET = 'yelpp'
else:
    print("[arg error!] please add at least one dataset argument")
    exit()
    
# prep-version argument parsing
if parser.parse_args().verone == True:
    PRPR_VER = 'verone'
elif parser.parse_args().vertwo == True:
    PRPR_VER = 'vertwo'
else:
    print("[arg error!] please add at least one preprocessing-version argument")
    exit()    
    
# word embedding argument parsing
if parser.parse_args().skip_word2vec_300d == True:
    WORD_EMBEDDING = 'skip_word2vec_300d'
elif parser.parse_args().glove_6b_300d == True:
    WORD_EMBEDDING = 'glove_6b_300d'
elif parser.parse_args().fasttext_300d == True:
    WORD_EMBEDDING = 'fasttext_300d'
else:
    print("[arg error!] please add at least one word-embedding argument")
    exit()    

# word embedding trainble argument parsing
if parser.parse_args().trainable == True:
    IS_TRAINABLE = True
elif parser.parse_args().nontrainable == True:
    IS_TRAINABLE = False
else:
    print("[arg error!] please add at least one trainable argument")
    exit() 
 
#################################################
""" PATH STETTING & MAKE DIRECTORIES """
#################################################   
### Path setting
ABSOLUTE_PATH = os.getcwd()
FULL_PATH = ABSOLUTE_PATH+'\\'+MODEL+'\\'+DATASET
#if EXP_NAME == '':
#    EXP_NAME = strftime("%Y-%m-%d_%Hh%Mm", gmtime()) # 명시안되어있으면 그냥 유니크값인 시간으로 이름.
FULL_PATH = FULL_PATH + '\\' + WORD_EMBEDDING+'___'+str(IS_TRAINABLE)+'___'+PRPR_VER + '\\'

# Full path에 맞게 폴더를 모두 생성. (recursive하게 모두 생성됨.)
if not os.path.exists(os.path.dirname(FULL_PATH)):
    try:
        os.makedirs(os.path.dirname(FULL_PATH))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise    
    
############################
""" PREPROCESSING """
############################    
# open file to save our log.
orig_stdout = sys.stdout
f_description = open(FULL_PATH+'description.txt', 'w')
sys.stdout = f_description

## Load vocabulary
vocab_filename = 'dataset-description/'+DATASET+'/vocab_'+PRPR_VER+'.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())

## Load, Preprocessing, Vectorize Dataset
Xtrain, ytrain, X_test, y_test, tokenizer = data_preprocessing(
                                            DATASET,
                                            PRPR_VER,
                                            vocab,
                                            PATH_SAMPLE,
                                            PATH_TEST,
                                            True) # is sample true or false?

# # split train and validation set
# # [주의]: cross-validation을 사용하면 아래 변수들은 그냥 사용안하면 된다.
# nb_validation_samples = int(VALIDATION_SPLIT * Xtrain.shape[0])
# X_train = Xtrain[:-nb_validation_samples]
# y_train = ytrain[:-nb_validation_samples]
# X_val = Xtrain[-nb_validation_samples:]
# y_val = ytrain[-nb_validation_samples:]
# #print(X_val.shape)

######################################
""" LOAD PRETRAINED WORD EMBEDDING """
######################################
embedding_matrix = load_pretrained_embedding(
                                    tokenizer.word_index, 
                                    WORD_EMBEDDING,
                                    DATASET, # pickle 이름때문에..
                                    PRPR_VER, # pickle 이름때문에..
                                    EMBEDDING_DIM)    
    
    
######################################
""" DEFINE MODEL """
######################################    
max_length = len(Xtrain[0])
vocab_size = len(tokenizer.word_index) + 1
nb_filter = 128

embedding_layer = Embedding(vocab_size, # vocab_size
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=max_length, # cnn모델에만 들어가는 것 확인!
                            trainable=IS_TRAINABLE) # is trainable?
convs = []
filter_sizes = [3,4,5]
sequence_input = Input(shape=(max_length,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

for fsz in filter_sizes:
    l_conv = Conv1D(filters=nb_filter,
                    kernel_size=fsz,
                    activation='relu')(embedded_sequences)
    l_pool = MaxPooling1D(5)(l_conv)
    convs.append(l_pool)
    
#l_merge = Merge(mode='concat', concat_axis=1)(convs) # old version
l_merge = concatenate(convs,  axis=1) # new version
l_cov1= Conv1D(nb_filter, 5, activation='relu')(l_merge)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(nb_filter, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(30)(l_cov2)
l_flat = Flatten()(l_pool2)
l_drop = Dropout(0.5)(l_flat)
l_dense = Dense(nb_filter, activation='relu')(l_drop)
preds = Dense(NUM_CLASSES, activation='softmax')(l_dense)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
print('\n')
print('[ MODEL SUMMARY ]')
print(model.summary())

# # 모델 정보 저장하기 (sumamry()와 비슷함)
# # serialize model to JSON
# model_json = model.to_json()
# with open(FULL_PATH + "model.json", "w") as json_file:
#     json_file.write(model_json)     
    
######################################
""" TRAIN AND EVALUATE MODEL """
######################################    
kfold = StratifiedKFold(n_splits=NB_FOLDS, shuffle=True, random_state=seed)
y_temp = [0] * len(Xtrain) # temporal variable
cv_scores = []
cnt = 1

#[주의]: Xtrain이 numpy arr이어야만 Xtrain[[1,6,8,9,11,...]]와 같은 필터링이 가능하다.
for train_idx_arr, val_idx_arr in kfold.split(Xtrain, y_temp): # y는 1dim이어야 하므로 여기서 임시로 1dim의 y_temp사용.
    
    if USE_VAL_SET == False: 
        # 마치 StratifiedKFold를 쓰지 않은 것 처럼. 그냥 전체 index를 할당함.
        # 따라서, 이 for문의 의미는 단지 횟수로만 의미가 있음.
        train_idx_arr = np.arange(len(Xtrain))
    
    print('\n\n********************* ', cnt, ' - TRAINING START *********************')

    # early stopping 정의
    #early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    # 실제 epoch=300으로 하면, early-stopping 조건에 맞으면 300 훨씬 전에도 끝날 수 있음
    # model.fit(,.., callbacks=[early_stopping])    
    
    if USE_VAL_SET == True:
        #[주의] panda dataframe 사용시에 .values를 해줘야 된다
        history = model.fit(Xtrain[train_idx_arr], 
                      ytrain.values[train_idx_arr], 
                      validation_data=(Xtrain[val_idx_arr], ytrain.values[val_idx_arr]),
                      epochs=NB_EPOCHS, 
                      batch_size=BATCH_SIZE,
                      verbose=2)
    elif USE_VAL_SET == False:
        history = model.fit(Xtrain[train_idx_arr], 
                      ytrain.values[train_idx_arr], 
                      #validation_data=(Xtrain[val_idx_arr], ytrain.values[val_idx_arr]),
                      epochs=NB_EPOCHS, 
                      batch_size=BATCH_SIZE,
                      verbose=2)
        
    print('\n[ HISTORY DURING TRAINING ]')
    print('\nhistory[loss]')
    print(history.history['loss'])
    print('\nhistory[acc]')
    print(history.history['acc'])
    if USE_VAL_SET == True:
        print('\nhistory[val_loss]')
        print(history.history['val_loss'])
        print('\nhistory[val_acc]')
        print(history.history['val_acc'])



    #[주의]: r_f file은 꼭 append mode ('a')로 open해야 한다.
    with open(FULL_PATH+'results.txt', 'a') as r_f:
        r_f.write('\n\n********************* '+ str(cnt)+' - EVALUATION START *********************')    
    
        ## Accuracy
        r_f.write('\n[ ACCURACY ]')
        #_, acc1 = model.evaluate(X_train, y_train.values, verbose=0)
        #print('Train Accuracy: %f' % (acc1*100))
        _, acc2 = model.evaluate(X_test, y_test.values, verbose=0)
        cv_scores.append(acc2*100)
        r_f.write('\n*Test Accuracy: ' + str(acc2*100))

        ## Classification Report
        yhat = model.predict(X_test, verbose=0)
        y_hat = list(map(argmax, yhat))
        y_true = [np.where(r==1)[0][0] for r in y_test.values]
        r_f.write('\n')
        r_f.write('\n[ MICRO-AVERAGED SCORE ]')
        r_f.write('\n   precision:\t\t' + str(metrics.precision_score(y_true, y_hat, average='micro')))
        r_f.write('\n   recall:\t\t' + str(metrics.recall_score(y_true, y_hat, average='micro')))
        r_f.write('\n   f1-score:\t\t' + str(metrics.f1_score(y_true, y_hat, average='micro')))
        r_f.write('\n')
        r_f.write('\n[ MACRO-AVERAGED SCORE ]')
        r_f.write('\n   precision:\t\t' + str(metrics.precision_score(y_true, y_hat, average='macro')))
        r_f.write('\n   recall:\t\t' + str(metrics.recall_score(y_true, y_hat, average='macro')))
        r_f.write('\n   f1-score:\t\t' + str(metrics.f1_score(y_true, y_hat, average='macro')))
        r_f.write('\n')
        r_f.write(classification_report(y_true, y_hat))    
    
    ## break point
    if cnt == 1: # cnt=1부터 시작
        # serialize weights to HDF5
        # 모델 저장은 한 번만 하자. (어짜피 다 똑같으니..)
        model.save_weights(FULL_PATH + "model.h5")
        cnt+=1
    elif cnt == NB_REAL_EX:
        # 끝내기 전에 모든 성능의 평균, 최대값을 제목으로 파일저장하자 (파일 내용은 자세히 모두 들어감)
        summary_result_file_name = '(avg)'+str(round(np.mean(cv_scores),3))+'__(max)'+str(round(np.max(cv_scores),3))
        with open(FULL_PATH+summary_result_file_name+'.txt', 'w') as sr_f:
            sr_f.write('Mean: ' + str(np.mean(cv_scores)))
            sr_f.write('\nMax: ' + str(np.max(cv_scores)))
            sr_f.write('\nMin: ' + str(np.min(cv_scores)))
            sr_f.write('\nstd: ' + str(np.std(cv_scores)))
            sr_f.write('\n')
            sr_f.write('\n[ACCURACY LIST (sv_scores list)]\n')
            for item in cv_scores:
                sr_f.write(str(item))
                sr_f.write(' ')
        break
    else: cnt+=1
        
# close files
sys.stdout = orig_stdout
f_description.close()    

print('\n>> '+MODEL+', '+WORD_EMBEDDING+', '+DATASET+', '+PRPR_VER+', '+str(IS_TRAINABLE)+' : Complete!\n')   
    
    
    
    
    
    
    
    
    