# Suitable Combination of Text Preprocessing and Word Embedding to Neural Network for Document Classification



## Prerequisites
We use Anaconda3-5.0.1-Linux-x86_64.sh. You can create a new vitual environment with all the dependencies in the yml files: 
`~$ conda env create -f environment-tensorflow-1.yml` or `environment-theano-1.yml`. You can check python libraries for this project in those .yml files.

## Dataset
For evaluation, we use AG's news dataset which is one of Zhang et al., 2015 Dataset. It was collected from AG's corpus of news article of the web and consists of the title and decroption fields with 4 classes. You can download it as csv format in [here](https://drive.google.com/open?id=1XbrUZk3_PFVEp7zkZVrNgnRRlXKgNWt3). 

## Usage

0. **Select Command**: for multiple programs at once, we select commands to be executed in `1_root_vocab.py` and `2_root_model.py` (arguments for core files (i.e. commands) are described in those files)

1. **Build Vocabulary** (according to (1)dataset and (2)preprocessing-type); data description is also saved.
```
~$ python 1_root_vocab.py
```
![](/assets/1_root_vocab.PNG)

2. **Run Model** (according to (1)model (2)dataset (3)preprocessing-type (4)word_embedding (5)is_trainable) 
```
~$ python 2_root_model.py
```
![](/assets/2_root_model.PNG)

## Contribution


## Summary




## Acknowledgement
Korea Institute of Science and Technology Information (KISTI) <br>
University of Science and Technology (UST) <br>
2017.11 ~ 2018.02
