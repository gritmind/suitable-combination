# Combinations of Text Preprocessing and Word Embedding Suitable for Neural Network Models for Document Classification 

최근 문서 분류를 위해 신경망 모델과 함께 워드 임베딩을 많이 사용한다. 특별한 이유없이 특정 워드 임베딩을 사용하거나 텍스트 전처리에 대한 명시를 보통 하지 않는 기존 연구들이 많다. 우리 연구는 신경망 모델의 성능을 높이는 한 가지 옵션으로 워드 임베딩과  텍스트 전처리의 적합한 조합을 제시한다. 추가적으로 패딩 방식과 미세조정에 의한 워드 임베딩 재학습 여부에 대한 분석을 실시한다.

![](/assets/process.PNG)

본 연구는 정보과학회논문지 제45권 7호에 "문서 분류를 위한 신경망 모델에 적합한 텍스트 전처리와 워드 임베딩의 조합"으로 게재(2018.07.15)되었음. [[논문](http://kiise.or.kr/e_journal/2018/7/JOK/pdf/08.pdf)] [[발표자료](https://1drv.ms/p/s!AllPqyV9kKUrkQD9pozYIUpXKd8q)]

## Prerequisites
We use Anaconda3-5.0.1-Linux-x86_64.sh. You can create a new vitual environment with all the dependencies in the yml files: 
`~$ conda env create -f environment-tensorflow-1.yml`  `environment-theano-1.yml`. You can check python libraries for this project in those .yml files.

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
* 텍스트 전처리와 워드 임베딩의 적합한(최적의) 조합 필요성 제기
* 조합 선정 기준 제시 (OOV단어 비율이 낮고, 어휘 사전 크기가 클수록 좋음)
* 구체적인 조합 제시 (구두점-분할, 레마타이징 & Skip-gram, GloVe.840B)
* 패딩 방식과 미세조정에 의한 워드 임베딩 재학습은 모델에 따라 달라짐
* 학습 데이터에 학습된 워드 임베딩 모델보다 사전에 학습된 워드 임베딩 모델의 우수성 확인
* 최신 모델 (K-CNN, Y-RNN, L-RCNN) 구현 및 문서 분류 모델 파이프라인(전처리-워드임베딩-모델) 구현
* 사전에 학습된 워드 임베딩 모델의 전처리와 태스크 전처리의 일치 여부 중요 & 추가적인 전처리 필요. (ex. 구두점 처리, 레마타이징)
* 딥러닝 환경 구축 에러 정리 [[here](https://github.com/gritmind/suitable-combination/blob/master/assets/2018-01-30-Error-Messages.md)]

## 
* 최근 신경망 모델을 사용하는 문서 분류 연구에서의 제한점 
   - 특별한 이유 없이 특정 워드 임베딩 모델만 사용
   - 명확히 명시되지 않은 텍스트 전처리 사용
* 성능 향상을 위한 옵션으로 워드 임베딩과 텍스트 전처리의 적합한 조합을 제시
   - OOV단어 비율, 어휘 사전 퀄리티 및 크기 관점으로 설명
* 비교 연구의 일반성을 위해 다양한 종류의 최신 모델들을 사용
   - 신경망 모델: CNN, RNN, RCNN
   - 워드 임베딩 모델: skip-gram, GloVe, fastText
* 추가로 패딩 방식과 미세조정에 의한 워드 임베딩 재학습 여부에 대한 비교 실험 실시





## Acknowledgement
Korea Institute of Science and Technology Information (KISTI) <br>
University of Science and Technology (UST) <br>
2017.11 ~ 2018.02
