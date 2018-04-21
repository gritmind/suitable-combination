# Combination of Text Preprocessing and Word Embedding suitable for Neural Network Models for Document Classification 

최근 문서 분류를 위해 신경망 모델과 함께 워드 임베딩을 많이 사용한다. 특별한 이유없이 특정 워드 임베딩을 사용하거나 텍스트 전처리에 대한 명시를 보통 하지 않는 기존 연구들이 많다. 우리 연구는 신경망 모델의 성능을 높이는 한 가지 옵션으로 워드 임베딩과 전처리의 적합한 조합을 제시한다. 추가적으로 패딩 방식과 미세조정에 의한 워드 임베딩 재학습 여부에 대한 분석을 실시한다.


본 연구는 논문으로도 작성됨: 문서 분류를 위한 신경망 모델에 적합한 텍스트 전처리와 워드 임베딩의 조합


## Prerequisites
We use Anaconda3-5.0.1-Linux-x86_64.sh. You can create a new vitual environment with all the dependencies in the yml files: 
`~$ conda env create -f environment-tensorflow-1.yml` or `environment-theano-1.yml`. You can check python libraries for this project in those .yml files.

## Dataset
For evaluation, we use AG's news dataset which is one of Zhang et al., 2015 Dataset. It was collected from AG's corpus of news article of the web and consists of the title and decroption fields with 4 classes. You can download it as csv format in [here](https://drive.google.com/open?id=1XbrUZk3_PFVEp7zkZVrNgnRRlXKgNWt3). 

## Pre-trained Word Embedding Model
Download below to `../embedding/skip-gram/`, `../glove.6B`, `../glove.42B/`, `../glove.840B/`, `../fastText/`, respectively. 
* Skip-gram (GoogleNews-vectors-negative300.bin) [[download](https://code.google.com/archive/p/word2vec/)]
* GloVe (glove.6B.300d.txt, glove.42B.300d.txt, glove.840B.300d.txt) [[download](https://nlp.stanford.edu/projects/glove/)]
* fastText (wiki.en.vec) [[download](https://fasttext.cc/docs/en/pretrained-vectors.html)]


## Usage

0. **Select Command**: for multiple programs at once, we select commands to be executed in `1_root_vocab.py` and `2_root_model.py` (arguments for core files (i.e. commands) are described in those files)

1. **Build Vocabulary** (according to (1)dataset and (2)preprocessing-type); data description is also saved.
```
~$ python 1_root_vocab.py
```
![](/assets/1_root_vocab2.PNG)

2. **Run Model** (according to (1)model (2)dataset (3)preprocessing-type (4)word_embedding (5)is_trainable) 
```
~$ python 2_root_model.py
```
![](/assets/2_root_model2.PNG)

* (Experimental Results): Below files will be saved at ../src/model#/dataset#/ 
   * results.txt: accuracy, confusion matrix
   * decription.txt: model summary, history(loss,acc) during training
   * model.h5: trained model is saved to HDF5
   * (avg)(max).txt: mean, max, min, std, avg accuracy 

![](/assets/3_result.PNG)


## Contribution


## Summary




## Acknowledgement
Korea Institute of Science and Technology Information (KISTI) <br>
University of Science and Technology (UST) <br>
2017.11 ~ 2018.02
