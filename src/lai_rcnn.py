
### EXPLICIT PARAMETER
##########################################
MODEL = 'lai_rcnn' 
IS_SAMPLE = False
EMBEDDING_DIM = 300 

""" Parameters """   
C_DIM = 300 # context vector dimension (same as word embedding dim)
Y_DIM = 100 # final representation dimension (same as hidden layer size = 100 (cf. paper))

NB_REAL_EX = 5 

BATCH_SIZE = 256   
NB_EPOCHS = 50

USE_VAL_SET = True 
PATIENCE = 10
VALIDATION_SPLIT = 0.1

from data_handler import *
from keras import backend
from keras.layers import Dense, Input, Lambda, LSTM, TimeDistributed
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
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
#seed = 7
#np.random.seed(seed)
parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")

####################################
""" PARSING STETTING """
####################################

# dataset
parser.add_argument('--agnews', action='store_true')
parser.add_argument('--yelpp', action='store_true')

# preprocessing version
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

# word embedding
parser.add_argument('--skip', action='store_true')
parser.add_argument('--glove_6b', action='store_true')
parser.add_argument('--glove_42b', action='store_true')
parser.add_argument('--glove_840b', action='store_true')
parser.add_argument('--fast', action='store_true')
parser.add_argument('--rand', action='store_true')
parser.add_argument('--average', action='store_true')
parser.add_argument('--gensim', action='store_true')
parser.add_argument('--gensimfast', action='store_true')

# word-embedding trainable check
parser.add_argument('--train', action='store_true')
parser.add_argument('--untrain', action='store_true')


# dataset argument parsing
if parser.parse_args().agnews == True: DATASET = 'agnews'
elif parser.parse_args().yelpp == True: DATASET = 'yelpp'
else:
    print("[arg error!] please add at least one dataset argument")
    exit()
    
# prep-version argument parsing
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
    print("[arg error!] please add at least one preprocessing-version argument")
    exit()

# word embedding argument parsing
if parser.parse_args().skip == True: WORD_EMBEDDING = 'skip_word2vec_300d'
elif parser.parse_args().glove_6b == True: WORD_EMBEDDING = 'glove_6b_300d'
elif parser.parse_args().glove_42b == True: WORD_EMBEDDING = 'glove_42b_300d'
elif parser.parse_args().glove_840b == True: WORD_EMBEDDING = 'glove_840b_300d'	
elif parser.parse_args().fast == True: WORD_EMBEDDING = 'fasttext_300d'
elif parser.parse_args().rand == True: WORD_EMBEDDING = 'rand_300d'
elif parser.parse_args().average == True: WORD_EMBEDDING = 'average_300d'
elif parser.parse_args().gensim == True: WORD_EMBEDDING = 'gensim_skip_300d'
elif parser.parse_args().gensimfast == True: WORD_EMBEDDING = 'gensim_fast_300d'
else:
    print("[arg error!] please add at least one word-embedding argument")
    exit()

# word embedding trainble argument parsing
if parser.parse_args().train == True: IS_TRAINABLE = True
elif parser.parse_args().untrain == True: IS_TRAINABLE = False
else:
    print("[arg error!] please add at least one trainable argument")
    exit() 

#################################################
""" PATH STETTING & MAKE DIRECTORIES """
#################################################   
### Path setting
ABSOLUTE_PATH = os.getcwd()
FULL_PATH = os.path.join(ABSOLUTE_PATH, MODEL, DATASET)
#if EXP_NAME == '':
#    EXP_NAME = strftime("%Y-%m-%d_%Hh%Mm", gmtime()) 
FULL_PATH = os.path.join(FULL_PATH, WORD_EMBEDDING+'___'+str(IS_TRAINABLE)+'___'+PRPR_VER, "")

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

## Model hyper-parameters
print('Batch size: ', BATCH_SIZE)
print('Epoach size: ', NB_EPOCHS)

## Load vocabulary
vocab_filename = os.path.join('dataset-description', DATASET, 'vocab_'+PRPR_VER+'.txt')
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())

## Load, Preprocessing, Vectorize Dataset
Xtrain, ytrain, X_test, y_test, tokenizer, len_info = data_preprocessing(
                                                                        MODEL,
                                                                        DATASET,
                                                                        PRPR_VER,
                                                                        vocab,
                                                                        IS_SAMPLE) # is sample true or false?                                                                     
VOCA_SIZE = len(tokenizer.word_index) + 1
NUM_CLASSES = len_info[0] 

######################################
""" LOAD PRETRAINED WORD EMBEDDING """
######################################
if WORD_EMBEDDING == 'average_300d':
    temp_tensor = []
    for emb in ['skip_word2vec_300d', 'glove_6b_300d', 'fasttext_300d']:
        temp_matrix = load_pretrained_embedding(
                                            tokenizer.word_index, 
                                            emb,
                                            DATASET, # pickle 
                                            PRPR_VER, # pickle
                                            EMBEDDING_DIM,
                                            IS_SAMPLE,
                                            vocab) 
        temp_tensor.append(temp_matrix)
    embedding_matrix = np.array(list(map(lambda x:sum(x)/float(len(x)), zip(*temp_tensor))))

else:
    embedding_matrix = load_pretrained_embedding(
                                        tokenizer.word_index, 
                                        WORD_EMBEDDING,
                                        DATASET, # pickle 
                                        PRPR_VER, # pickle
                                        EMBEDDING_DIM,
                                        IS_SAMPLE,
                                        vocab)    

######################################
""" DEFINE MODEL """
######################################    

def define_rcnn_model(VOCA_SIZE, cnt):

    org_word_seq = Input(shape = (None, ), dtype = "int32")
    left_context_seq = Input(shape = (None, ), dtype = "int32")
    right_context_seq = Input(shape = (None, ), dtype = "int32")

    if WORD_EMBEDDING == 'rand_300d':
        embedder = Embedding(VOCA_SIZE, # vocab_size
                            EMBEDDING_DIM,
                            embeddings_initializer = 'uniform',
                            trainable=IS_TRAINABLE) # is trainable?    
    
    else: # use pretrained word embedding
        embedder = Embedding(VOCA_SIZE, # vocab_size
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=IS_TRAINABLE)

    word_embed_seq = embedder(org_word_seq)
    left_context_emb_seq = embedder(left_context_seq)
    rght_context_emb_seq = embedder(right_context_seq)

    # Word Representation
    left_context_vector = LSTM(C_DIM, return_sequences = True)(left_context_emb_seq)
    right_context_vector = LSTM(C_DIM, return_sequences = True, go_backwards = True)(rght_context_emb_seq)
    concat_all = concatenate([left_context_vector, word_embed_seq, right_context_vector], axis = 2)
    latent_semantic_vector = TimeDistributed(Dense(Y_DIM, activation = "tanh"))(concat_all)

    # Text Representation
    max_pool_output_vector = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (Y_DIM, ))(latent_semantic_vector)
    output = Dense(NUM_CLASSES, input_dim = Y_DIM, activation = "softmax")(max_pool_output_vector)

    model = Model(inputs = [org_word_seq, left_context_seq, right_context_seq], outputs = output)
    model.compile(loss = "categorical_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])
    
    if cnt==1:
        print('\n')
        print('[ MODEL SUMMARY ]')
        print(model.summary())

    return model
    
######################################
""" TRAIN AND EVALUATE MODEL """
######################################    

# split train and validation set
nb_validation_samples = int(VALIDATION_SPLIT * Xtrain.shape[0])
X_train = Xtrain[:-nb_validation_samples]
y_train = ytrain[:-nb_validation_samples]

X_val = Xtrain[-nb_validation_samples:]
y_val = ytrain[-nb_validation_samples:]
#print(X_val.shape)

### PREPARRING DATASET FOR TRAINING
def shift_right(doc): return [VOCA_SIZE-1] + list(doc[:-1])
def shift_left(doc): return list(doc[1:]) + [VOCA_SIZE-1]

if USE_VAL_SET == False: 
    left_Xtrain = np.array(list(map(shift_right, Xtrain)))
    right_Xtrain = np.array(list(map(shift_left, Xtrain)))   
    
elif USE_VAL_SET == True:  
    left_X_train = np.array(list(map(shift_right, X_train)))
    right_X_train = np.array(list(map(shift_left, X_train)))
    left_X_val = np.array(list(map(shift_right, X_val)))
    right_X_val = np.array(list(map(shift_left, X_val)))

left_X_test = np.array(list(map(shift_right, X_test)))
right_X_test = np.array(list(map(shift_left, X_test)))

cv_scores = []
cnt = 1

while(True):
 
    # Load pre-defined Model
    model = define_rcnn_model(VOCA_SIZE, cnt)

    print('\n\n********************* ', cnt, ' - TRAINING START *********************')
    
    if USE_VAL_SET == False: 
        history = model.fit(
                        [Xtrain, left_Xtrain, right_Xtrain], 
                        ytrain,
                        #validation_data=(
                        #    [Xtrain[val_idx_arr], left_Xtrain[val_idx_arr], right_Xtrain[val_idx_arr]], 
                        #    ytrain.values[val_idx_arr]),
                        batch_size = BATCH_SIZE,
                        epochs = NB_EPOCHS,
                        verbose=2)       
    
    elif USE_VAL_SET == True:    
        early_stopping = EarlyStopping(monitor='val_acc', patience=PATIENCE, verbose=0)
        history = model.fit(
                        [X_train, left_X_train, right_X_train], 
                        y_train,
                        validation_data=(
                            [X_val, left_X_val, right_X_val], 
                            y_val),
                        batch_size = BATCH_SIZE,
                        epochs = NB_EPOCHS,
                        callbacks=[early_stopping],
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

    with open(FULL_PATH+'results.txt', 'a') as r_f:
        r_f.write('\n\n********************* '+ str(cnt)+' - EVALUATION START *********************')    

        ## Accuracy
        r_f.write('\n[ ACCURACY ]')
        #_, acc1 = model.evaluate(X_train, y_train, verbose=0)
        #print('Train Accuracy: %f' % (acc1*100))
        _, acc2 = model.evaluate([X_test, left_X_test, right_X_test], y_test, verbose=0)
        cv_scores.append(acc2*100)
        r_f.write('\n*Test Accuracy: '+str(acc2*100))

        ## Classification report
        yhat = model.predict([X_test, left_X_test, right_X_test], verbose=0)
        y_hat = list(map(argmax, yhat))
        y_true = [np.where(r==1)[0][0] for r in y_test]
        r_f.write('\n')
        r_f.write('\n[ MICRO-AVERAGED SCORE ]')
        r_f.write('\n   precision:\t\t'+str(metrics.precision_score(y_true, y_hat, average='micro')))
        r_f.write('\n   recall:\t\t'+str(metrics.recall_score(y_true, y_hat, average='micro')))
        r_f.write('\n   f1-score:\t\t'+str(metrics.f1_score(y_true, y_hat, average='micro')))
        r_f.write('\n')
        r_f.write('\n[ MACRO-AVERAGED SCORE ]')
        r_f.write('\n   precision:\t\t'+str(metrics.precision_score(y_true, y_hat, average='macro')))
        r_f.write('\n   recall:\t\t'+str(metrics.recall_score(y_true, y_hat, average='macro')))
        r_f.write('\n   f1-score:\t\t'+str(metrics.f1_score(y_true, y_hat, average='macro')))
        r_f.write('\n')
        r_f.write(classification_report(y_true, y_hat))    
     
    # serialize weights to HDF5
    model.save_weights(FULL_PATH + str(cnt)+"___model.h5"+'___'+str(IS_TRAINABLE)+'___'+WORD_EMBEDDING+'___'+PRPR_VER)
    
    if cnt == NB_REAL_EX:
        summary_result_file_name = '(avg)'+str(round(np.mean(cv_scores),3))+'__(max)'+str(round(np.max(cv_scores),3))
        with open(FULL_PATH+summary_result_file_name+'.txt', 'w') as sr_f:
            sr_f.write('Mean: '+str(np.mean(cv_scores)))
            sr_f.write('\nMax: '+str(np.max(cv_scores)))
            sr_f.write('\nMin: '+str(np.min(cv_scores)))
            sr_f.write('\nstd: '+str(np.std(cv_scores)))
            sr_f.write('\n')
            sr_f.write('\n[ACCURACY LIST(sv_scores list)]\n')
            for item in cv_scores:
                sr_f.write(str(item))
                sr_f.write(' ')
        break
        
    cnt+=1
        
# close files
sys.stdout = orig_stdout
f_description.close()

print('\n\n>> '+MODEL+', '+WORD_EMBEDDING+', '+DATASET+', '+PRPR_VER+', '+str(IS_TRAINABLE)+' : Complete! (from main.py)\n\n')
