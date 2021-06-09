## 텐서(Tensor)
* 0차원 데이터 : 스칼라
* 1차원의 순서가 있는 데이터의 묶음 : 벡터
* 2차원의 '행'과 '열'이 있는 데이터의 묶음 : 행렬
* 그 이후의 차원 : **텐서**
</br></br>
## 텐서플로우(TensorFlow)
* 데이터 플로우 그래프를 쉽게 만들고 실행할 수 있도록 도와주는 라이브러리
* 텐서플로우를 이용하여 머신러닝 수행 시 3단계를 거친다.
  - 1.데이터 플로우 그래프를 빌드(build)하는 단계
  - 2.데이터를 입력하고 그래프를 실행(run)하는 단계
  - 3.그래프 내부 변수들을 업데이트(update)하고 출력값을 return하는 단계
</br></br>
## 그래프
* 노드(node)나 꼭지점(vertex)로 연결 되어 있는 개체(entity)의 집합을 부르는 용어
* tensorflow에서 그래프의 각 노드는 하나의 연산을 나타내며, 입력값을 받아 다른 노드로 전달할 결과를 출력
![1](https://user-images.githubusercontent.com/84856055/120320658-6459b280-c31d-11eb-9b3f-407daaaaaf56.JPG)
</br></br>
## tf.constant(value, dtype, shape, name)
### 결과값은 value 인자와 shape에 의해 결정됨으로써 dtype의 값으로 채워짐
* value = 상수 또는 dtype 타입을 가진 값들의 리스트
* shape = 인자가 존재할 경우 차원을 결정, **그 외에는 value의 shape를 그대로 사용**
* dtype = 인자가 존재하지 않을 경우, **value로 부터 타입을 추론하여 사용**
* name = 텐서의 명칭      
```python3
import tensorflow as tf

a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a,b)
e = tf.add(c,b)
f = tf.subtract(d,e)
with tf.Session() as sess:
    feches = [d,e,f]
    outs = sess.run(feches)
```

![2](https://user-images.githubusercontent.com/84856055/120320668-66237600-c31d-11eb-9932-f8ab32f6d249.JPG)
</br></br>
## tf.random_normal(shape, mean = 0.0, stddev = 1.0)
### 정규분포로부터 난수값을 반환
 * shape = 정수값의 tensor 또는 python 배열
 * mean = 정규분포의 평균값
 * sttdev = 정규분포의 표준 편차   
```python3
init_v = tf.random_normal((1, 5), 0, 1) # shape = (1x5), mean = 0, sttdev = 1
```
</br></br>
## tf.Session()
### operation 객체를 실행하고, tensor 객체를 평가하기 위한 환경을 제공하는 객체   
![session](https://user-images.githubusercontent.com/84856055/120320676-6885d000-c31d-11eb-8ad3-2f1645820ea1.JPG)
</br>
![session-3](https://user-images.githubusercontent.com/84856055/120320685-6a4f9380-c31d-11eb-82b9-0d8e99944c1b.JPG)   
session **실행 전 실제 tensor(data)는 흐르지 않으며, 연산 또한 수행되지 않는다.**   
session **실행은 session.run() 함수를 이용**, ()안에 tensor나 연산을 넣어줄 수 있다.    
<br>
![session-2](https://user-images.githubusercontent.com/84856055/120320694-6b80c080-c31d-11eb-83dc-89b0a8052f30.JPG)
</br>
##### session의 실행 모습 전과 후 비교
</br></br>
## Dense 레이어 (순차형 모델 Sequential 기준으로 일단 작성, 차후에 추가)
### 입력과 출력을 모두 연결해주며 입력과 출력을 각각 연결해주는 가중치를 포함하고 있다.
```python3
Dense(5, input_dim = 2, activation = 'sigmoid')
```
* 첫번째 인자 : 출력 뉴런(노드)의 수를 결정, 이 층에 5개의 노드를 만들겠다는 뜻
* 두번째 인자 : 입력 뉴런(노드)의 수를 결정, 입력 데이터에서 몇개의 값을 가져올지 정하는 것으로 **맨 처음 입력층에서만 사용**
* 세번째 인자 : 활성화 함수를 선택   
  - linear : default 값으로 입력 뉴런과 가중치로 계산된 결과값이 그대로 출력
  - relu : 은닉층으로 학습, 역전파를 통해 좋은 성능이 나오기 때문에 마지막 층이 아니면 대부분 relu 를 이용
  - sigmoid : Yes or No 와 같은 **이진 분류 문제**에 사용, 출력층에 주로 쓰임
  - softmax : 확률 값을 이용해 **다양한 클래스를 분류**하기 위한 문제에 사용, 출력층에 주로 쓰임
```python3
model = Sequential()
model.add(Dense(5, input_dim = 1, activation = '활성화 함수')) # 첫번째 Dense가 은닉층 + 입력층의 역할을 겸함
model.add(Dense(3, activation = '활성화 함수')) # 은닉층
model.add(Dense(1, activation = '활성화 함수')) # 맨 마지막 층은 결과를 출력하는 출력층의 역할
```
![2](https://user-images.githubusercontent.com/84856055/120343346-96760f00-c333-11eb-883a-bb137ec9a868.JPG)
<br><br>
## Compile
### 만들어진 모델을 컴파일, 학습에 대한 설정
```python3
model.compile(loss = 'binary_crossentropy', optimizer = 'SGD', metrics=['accuracy'])
```
* loss : 손실함수를 설정해주는 부분, 한 번 신경망이 실행될 때마다 오차 값을 추적하는 함수 (참조 : <https://keras.io/losses/>
  - binary_crossentropy : 이진 분류
  - categorical_crossentropy : 일반적인 분류<br>
  - **Dense 레이어에서 사용되는 손실함수와 다르다**
* optimizer : 최적화 함수를 설정하는 부분, 오차를 어떻게 줄여 나갈지 정하는 함수 (참조 : <https://keras.io/ko/optimizers/>)
* metrics : 모델 수행 결과를 나타내게끔 설정하는 부분으로 과적합을 방지하는 기능 (참조 : <https://keras.io/ko/metrics/>)
<br><br>
## fit
### 컴파일한 모델을 훈련
```python3
model.fit(x_data, y_data, epochs = 1000, batch_size = 1)
```
* 첫번째 인자 : 입력 데이터
* 두번째 인자 : 출력 데이터
* epochs : 훈련 횟수
* batch_size : 작업단위를 의미, default는 32 (**너무 크면 학습 속도가 느려지고, 너무 작으면 실행 값의 편차가 생겨 결과값이 불안정**)
<br><br>
## Dummy Variable (더미변수)
* 카테고리형 데이터(Categorical Data)를 수치형 데이터(Numerical Data)로 변환한 데이터를 뜻함
* 카테고리형 데이터의 경우 일반적으로 회귀분석과 같은 연속형 변수를 다루는 분석기법에서는 사용할 수 없기 때문에 수치형 데이터로 변환해 주어야 함
<br><br>
## 데이터 전처리
### 머신러닝 알고리즘은 문자열 값을 입력 값으로 허락하지 않기 때문에 모든 문자열 값들을 숫자형으로 인코딩하는 전처리 작업(Preprocessing) 후에 모델에 학습을 시켜야 함<br>
1. LabelEncoder : 문자를 0부터 시작하는 정수형 숫자로 바꿔주는 기능, 반대로 코드숫자를 이용하여 원본 값 구하기 가능
  - 일괄적인 숫자 값으로 변환되면서 예측 성능이 떨어질 수 있음
  - 선형 회귀와 같은 알고리즘에는 적용하지 않아야 함 but, 트리 계열의 알고리즘은 숫자의 이러한 특성을 반영하지 않으므로 가능
![labelencoder](https://user-images.githubusercontent.com/84856055/120481004-a00e7e00-c3ea-11eb-84ab-cbced65da799.JPG)
#### Label Encoder 예시
![1](https://user-images.githubusercontent.com/84856055/121134916-ee020680-c86e-11eb-8331-87767e1295a6.JPG)    
#### 다른 방법 예시
<br><br>
2. One-hot encoding : 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식
  - 각 단어에 고유한 index를 부여 (정수 인코딩)
  - 표현하고 싶은 단어의 index 위치에 1을 부여, 다른 단어의 index 위치에 0을 부여
![result](https://user-images.githubusercontent.com/84856055/120481579-3478e080-c3eb-11eb-9aec-b1c08675d730.JPG)   
  - list 내의 원소들 One-hot encoding 방식 : **빈 list에 하나씩 넣어서 해당 list 통째로 넣어줌**<br>
![1](https://user-images.githubusercontent.com/84856055/121154828-3d9dfd80-c882-11eb-8cb4-b9744d1c9570.JPG)<br>
![one-hot encoding](https://user-images.githubusercontent.com/84856055/120481051-abfa4000-c3ea-11eb-9131-f533a732cbac.JPG)
#### Label encoding과 One-hot encoding 차이<br>
