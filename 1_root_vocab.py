# -*- coding: utf-8 -*-
# to build vocubulary based on specific dataset and preprocessing
import sys
import os

os.chdir('src') 
command_str_list = [ 

    ###################################
    ### dataset: AG-news
    ###################################
    
    #"python build_vocab.py --agnews --ver_a",
    "python build_vocab.py --agnews --ver_b",
    #"python build_vocab.py --agnews --ver_c", 
    #"python build_vocab.py --agnews --ver_d",
    #"python build_vocab.py --agnews --ver_e",
    "python build_vocab.py --agnews --ver_f",
    #"python build_vocab.py --agnews --ver_g",
    #"python build_vocab.py --agnews --ver_h",
    #"python build_vocab.py --agnews --ver_i",
    #"python build_vocab.py --agnews --ver_j",
    "python build_vocab.py --agnews --ver_k",
    "python build_vocab.py --agnews --ver_l",
    #"python build_vocab.py --agnews --ver_m",
    "python build_vocab.py --agnews --ver_n",
    #"python build_vocab.py --agnews --ver_o",
    #"python build_vocab.py --agnews --ver_p",
    "python build_vocab.py --agnews --ver_q"
    #"python build_vocab.py --agnews --ver_r"
    
]

for command in command_str_list:
    os.system(command)
    print('\n\n       Done! (from root.py): ' + command + '\n\n')

os.chdir('..')
