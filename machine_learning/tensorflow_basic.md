## 그래프
* 노드(node)나 꼭지점(vertex)로 연결 되어 있는 개체(entity)의 집합을 부르는 용어
* tensorflow에서 그래프의 각 노드는 하나의 연산을 나타내며, 입력값을 받아 다른 노드로 전달할 결과를 출력
![1](https://user-images.githubusercontent.com/84856055/120320658-6459b280-c31d-11eb-9b3f-407daaaaaf56.JPG)

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

## tf.Session()
### operation 객체를 실행하고, tensor 객체를 평가하기 위한 환경을 제공하는 객체   
![session](https://user-images.githubusercontent.com/84856055/120320676-6885d000-c31d-11eb-8ad3-2f1645820ea1.JPG)    
![session-3](https://user-images.githubusercontent.com/84856055/120320685-6a4f9380-c31d-11eb-82b9-0d8e99944c1b.JPG)
### session 실행 전 실제 tensor(data)는 흐르지 않으며, 연산 또한 수행되지 않는다.
### session의 실행은 sess.run() 함수를 이용, ()안에 tensor나 연산을 넣어줄 수 있다 
![session-2](https://user-images.githubusercontent.com/84856055/120320694-6b80c080-c31d-11eb-83dc-89b0a8052f30.JPG)
#### session의 실행 모습 전과 후 비교    
