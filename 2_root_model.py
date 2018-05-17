# -*- coding: utf-8 -*-
#from main import *
import sys
import os

os.chdir('src') 
command_str_list = [ 

    """ Argment Description """
    ### arg1: dataset (e.g. agnews)
    ### arg2: type of preprocessing (e.g type a ~ l) 
    ### arg3: kind of word embedding (e.g. skip-gram, glove, fasttext)
    ### arg4: is word embedding trainable? (train vs. untrain)


    "python lai_rcnn.py --agnews --ver_f --glove_6b --train",
    "python lai_rcnn.py --agnews --ver_f --fast --train",
    "python lai_rcnn.py --agnews --ver_f --glove_42b --train",      
    
    "python lai_rcnn.py --agnews --ver_k --skip --train",
    "python lai_rcnn.py --agnews --ver_k --glove_840b --train",

    "python lai_rcnn.py --agnews --ver_n --glove_6b --train",
    "python lai_rcnn.py --agnews --ver_n --fast --train",
    "python lai_rcnn.py --agnews --ver_n --glove_42b --train",
    "python lai_rcnn.py --agnews --ver_n --glove_840b --train",
    
    "python lai_rcnn.py --agnews --ver_q --skip --train",
    "python lai_rcnn.py --agnews --ver_q --glove_840b --train"
    ]

for command in command_str_list:
    os.system(command)
    print('\n\n       Done! (from root.py): ' + command + '\n\n')
os.chdir('..')

# cf. argparser example
#import argparse
#parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
#parser.add_argument('--w', action='store_true')
#args = parser.parse_args()
#print(args.w)
##python myparser.py -w