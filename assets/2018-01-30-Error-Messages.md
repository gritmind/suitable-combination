
## Environment Setting

딥러닝 실험환경을 구축하는데 있어서 생기는 오류의 종류는 크게 3가지로 구분됩니다.

1. VERSION (Python/Tensorflow/Theano/Keras/Cuda/Cudnn)

위 모든 버전이 각각 맞아야 함은 물론이고 서로 호환가능하여야 합니다. 예를 들어, 특정 tensorflow 버전은 특정 cuda와 cudnn 버전만 호환가능합니다. 
특정 keras 버전은 특정 tensorflow 버전과 호환가능합니다 (만약 tensorflow가 새로 업데이트 된 경우 keras와 곧바로 호환되지 않습니다) keras의 경우는 버전에 민감합니다. 특정 프로그램은 특정 케라스 버전에서만 동작하는 경우도 있습니다.
(2018-01-31기준: Python 3.5/ Tensorflow-gpu 1.4/ Keras 2.1.3/ Cuda 8.0/ cudnn 6.0)


2. PATH (Cuda/Anaconda)

path는 윈도우보다 리눅스에서 더 신경써야 합니다. (윈도우는 설치시에 자동으로 잡히는 것을 그대로 사용했습니다)

3. Encoding (UTF-8 Encoding)

utf-8 인코딩 문제은 python 2에서 자주 등장합니다. 간혹 버전이 맞지 않을 경우에도 등장하기도 합니다. 이런 경우 새롭게 버전에 맞게 설치하면 해결되는 경우도 있었습니다. python2-keras에서도 utf-8 인코딩 문제가 등장하는데, 구글링해서 keras.preprocessing에 있는 파일을 수정하면 됩니다. python3-keras에서 utf-8 인코딩문제가 발생했는데 한글주석을 지우니 사라졌습니다 (확실x).

conda를 사용하면 패키지화하는 것이 버젼간의 충돌을 막아줍니다. (global하게 사용하는 경우 맞지 않는 버전끼리 충돌할 가능성이 큽니다) 패키지화를 해야하는 또 다른 이유는 프로그램 단위로 요구하는 버전들이 다르기 때문입니다. 심하면 프로그램 단위로 conda 모듈을 만들어야 할지도 모릅니다
theano-gpu는 (반드시) conda로 설치하는 것이 좋습니다 (2018-01-31기준) (pip는 설치하는 과정도 많고, 오류도 많습니다)


## Error messages

`ValueError: unsupported pickle protocol: 3`
* 위 에러 메시지는 python 2에서 발생합니다 (python2에서는 pickle라이브러리보다는 cPickle이 더 적합합니다)
  * pickle을 사용하는 경우 dump할 때 protocol=2를 명시해줍니다
  * protocol=3로 저장한 pickle 파일을 load할 때 에러 메시지 발생합니다. 따라서, protocol=2로 pickle 파일을 재저장하고 다시 load해야 합니다.
