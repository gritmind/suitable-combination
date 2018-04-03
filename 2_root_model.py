import sys
import os

os.chdir('src') 
command_str_list = [ 

    """ Argment Description """
    ### arg1: dataset (e.g. agnews)
    ### arg2: type of preprocessing (e.g type a ~ l) 
    ### arg3: kind of word embedding (e.g. skip-gram, glove, fasttext)
    ### arg4: is word embedding trainable? (train vs. untrain)
    
    
    ###########################################################
    ##### Kim_CNN
    ###########################################################
    #"python kim_cnn.py --agnews --ver_e --skip --train", 
    #"python kim_cnn.py --agnews --ver_e --glove_42d --train", 
    #"python kim_cnn.py --agnews --ver_e --glove_840d --train", 
    #"python kim_cnn.py --agnews --ver_e --fast --train", 
    #"python kim_cnn.py --agnews --ver_e --rand --train", 
    #"python kim_cnn.py --agnews --ver_e --average --train", 
    #"python kim_cnn.py --agnews --ver_e --gensimfast --train", 
    
    #"python kim_cnn.py --agnews --ver_j --skip --train", 
    #"python kim_cnn.py --agnews --ver_j --glove --train", 
    #"python kim_cnn.py --agnews --ver_j --fast --train", 
    #"python kim_cnn.py --agnews --ver_j --rand --train", 
    #"python kim_cnn.py --agnews --ver_j --average --train", 
    #"python kim_cnn.py --agnews --ver_j --gensimfast --train", 
    
    #"python kim_cnn.py --agnews --ver_l --skip --train", 
    "python kim_cnn.py --agnews --ver_l --glove_840b --train" 
    #"python kim_cnn.py --agnews --ver_l --fast --train", 
    #"python kim_cnn.py --agnews --ver_l --rand --train", 
    #"python kim_cnn.py --agnews --ver_l --average --train", 
    #"python kim_cnn.py --agnews --ver_l --gensimfast --train",   
    
    
    ###########################################################
    ##### Yang_RNN
    ###########################################################
    #"python yang_rnn.py --agnews --ver_e --skip --train", 
    #"python yang_rnn.py --agnews --ver_e --glove --train", 
    #"python yang_rnn.py --agnews --ver_e --fast --train",
    #"python yang_rnn.py --agnews --ver_e --rand --train", 
    #"python yang_rnn.py --agnews --ver_e --average --train", 
    #"python yang_rnn.py --agnews --ver_e --gensimfast --train", 
    
    #"python yang_rnn.py --agnews --ver_j --skip --train", 
    #"python yang_rnn.py --agnews --ver_j --glove --train", 
    #"python yang_rnn.py --agnews --ver_j --fast --train",
    #"python yang_rnn.py --agnews --ver_j --rand --train", 
    #"python yang_rnn.py --agnews --ver_j --average --train", 
    #"python yang_rnn.py --agnews --ver_j --gensimfast --train", 
    
    #"python yang_rnn.py --agnews --ver_l --skip --train", 
    #"python yang_rnn.py --agnews --ver_l --glove --train",
    #"python yang_rnn.py --agnews --ver_l --fast --train", 
    #"python yang_rnn.py --agnews --ver_l --rand --train", 
    #"python yang_rnn.py --agnews --ver_l --average --train", 
    #"python yang_rnn.py --agnews --ver_l --gensimfast --train",  
    
    
    ###########################################################
    ##### Lai_RCNN
    ###########################################################
    #"python lai_rcnn.py --agnews --ver_e --skip --train", 
    #"python lai_rcnn.py --agnews --ver_e --glove --train", 
    #"python lai_rcnn.py --agnews --ver_e --fast --train", 
    #"python lai_rcnn.py --agnews --ver_e --rand --train", 
    #"python lai_rcnn.py --agnews --ver_e --average --train", 
    #"python lai_rcnn.py --agnews --ver_e --gensimfast --train", 
    
    #"python lai_rcnn.py --agnews --ver_j --skip --train", 
    #"python lai_rcnn.py --agnews --ver_j --glove --train", 
    #"python lai_rcnn.py --agnews --ver_j --fast --train", 
    #"python lai_rcnn.py --agnews --ver_j --rand --train", 
    #"python lai_rcnn.py --agnews --ver_j --average --train", 
    #"python lai_rcnn.py --agnews --ver_j --gensimfast --train", 
    
    #"python lai_rcnn.py --agnews --ver_l --skip --train", 
    #"python lai_rcnn.py --agnews --ver_l --glove --train", 
    #"python lai_rcnn.py --agnews --ver_l --fast --train", 
    #"python lai_rcnn.py --agnews --ver_l --rand --train", 
    #"python lai_rcnn.py --agnews --ver_l --average --train", 
    #"python lai_rcnn.py --agnews --ver_l --gensimfast --train" # end of commands: delete ','  
    
]

print('\n>>> RUN START: "2_root_model.py"\n')
for command in command_str_list:
    os.system(command)
    print('>> FINISH: "'+command+'"\n')

os.chdir('..')
