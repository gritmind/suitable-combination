import sys
import os

os.chdir('src') 

command_str_list = [ 

    """ Argment Description """
    ### arg1: dataset (e.g. agnews)
    ### arg2: type of preprocessing (e.g type a ~ l) 
    
    #"python build_vocab.py --agnews --ver_a",
    #"python build_vocab.py --agnews --ver_b",
    #"python build_vocab.py --agnews --ver_c",
    #"python build_vocab.py --agnews --ver_d",
    "python build_vocab.py --agnews --ver_e",
    #"python build_vocab.py --agnews --ver_f",
    #"python build_vocab.py --agnews --ver_g",
    #"python build_vocab.py --agnews --ver_h",    
    #"python build_vocab.py --agnews --ver_i",
    #"python build_vocab.py --agnews --ver_j",
    #"python build_vocab.py --agnews --ver_k",
    "python build_vocab.py --agnews --ver_l"  
]

print('\n>>> RUN START: "1_root_vocab.py"\n')
for command in command_str_list:
    os.system(command)
    print('>> FINISH: "'+command+'"\n')

os.chdir('..')
