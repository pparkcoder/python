## 텐서(Tensor)
* 0차원 데이터 : scaler
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
      
<pre>
<code>
a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a,b)
e = tf.add(c,b)
f = tf.subtract(d,e)
with tf.Session() as sess:
    feches = [d,e,f]
    outs = sess.run(feches)
</code>
</pre>   

![2](https://user-images.githubusercontent.com/84856055/120320668-66237600-c31d-11eb-9932-f8ab32f6d249.JPG)
</br></br>
## tf.random_normal(shape, mean = 0.0, stddev = 1.0)
### 정규분포로부터 난수값을 반환
 * shape = 정수값의 tensor 또는 python 배열
 * mean = 정규분포의 평균값
 * sttdev = 정규분포의 표준 편차   
<pre>
<code>
init_v = tf.random_normal((1, 5), 0, 1) # shape = (1x5), mean = 0, sttdev = 1
</code>
</pre>
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
<pre>
<code>
Dense(1, input_dim = 2, activation = 'sigmoid')
</code>
</pre>
* 첫번째 인자 : 출력 뉴런(노드)의 수를 결정
* 두번째 인자 : 입력 뉴런(노드)의 수를 결정, **맨 처음 입력층에서만 사용**
* 세번째 인자 : 활성화 함수를 선택   
  - linear : default 값으로 입력 뉴런과 가중치로 계산된 결과값이 그대로 출력
  - relu : 은닉층으로 학습, 역전파를 통해 좋은 성능이 나오기 때문에 마지막 층이 아니면 대부분 relu 를 이용
  - sigmoid : Yes or No 와 같은 **이진 분류 문제**에 사용, 출력층에 주로 쓰임
  - softmax : **확률 값을 이용**해 다양한 클래스를 분류하기 위한 문제에 사용, 출력층에 주로 쓰임
<pre>
<code>
model = Sequential()
model.add(Dense(5, input_dim = 1, activation = '활성화 함수')) 
model.add(Dense(3, activation = '활성화 함수'))
model.add(Dense(1, activation = '활성화 함수')) 
</code>
</pre>
![2](https://user-images.githubusercontent.com/84856055/120343346-96760f00-c333-11eb-883a-bb137ec9a868.JPG)
<br><br><br>
## Compile
### 만들어진 모델을 컴파일, 학습에 대한 설정
<pre>
<code>
model.compile(loss = 'binary_crossentropy', optimizer = 'SGD', metrics=['accuracy'])
</code>
</pre>
* loss : 손실함수를 설정해주는 부분 (참조 : <https://keras.io/losses/>
* optimizer : 최적화 함수를 설정하는 부분 (참조 : <https://keras.io/ko/optimizers/>)
* metrics : 모델의 성능을 판정하는데 사용하는 지표 (참조 : <https://keras.io/ko/metrics/>)
<br><br>
## fit
### 컴파일한 모델을 훈련
<pre>
<code>
model.fit(x_data, y_data, epochs = 1000, batch_size = 1)
</code>
</pre>
* 첫번째 인자 : 입력 데이터
* 두번째 인자 : 출력 데이터
* epochs : 훈련 횟수
* batch_size : 작업단위를 의미, default는 32
