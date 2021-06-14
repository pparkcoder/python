## CNN(Convolutional Neural Network)
 - 합성곱을 적용한 신경망으로서, 입력된 이미지에서 다시 한번 특징을 추출하기 위해 커널(슬라이딩 윈도우)을 도입하는 기법
 - 추출된 특징을 기반으로 기존의 뉴럴 네트워크를 이용하여 분류
 - **가장 중요한 것은 각 함수를 적용시 output의 크기를 아는것**<br><br>
![cnn 기본구조1](https://user-images.githubusercontent.com/84856055/121896715-92e37e80-cd5c-11eb-9858-f010b46adaa6.JPG)
<br><br>
### 합성곱(Convolution)
두 함수 f,g 가운데 하나의 함수를 반전(reverse), 전이(shift) 시킨 다음, 다른 하나의 함수와 곱한 결과를 적분
![1](https://user-images.githubusercontent.com/84856055/121897803-b3f89f00-cd5d-11eb-91e5-2e5469b9da57.jpg)
<br><br>
### 필터, 커널 (Filter = Kernel)
- 이미지의 특성을 찾아내기 위한 공용 파라미터로, 어떤 특징이 데이터에 있는지 없는지를 검출하는 역할
- 일반적으로 (4,4) 또는 (3,3)과 같은 정사각 행렬로 정의
- 입력 데이터를 지정된 간격으로 순회하며 합성곱을 통해서 만든 출력물 Feature Map을 만듬
```python3
model.add(Con2D(32, kernel_size = (3,3), input_shape = (28,28,1), activation = 'relu'))
```
 - 컨볼루션 레이어(합성곱 층)을 추가하는 함수
 - 첫번째 인자 : 적용할 커널의 수, 여기서는 32개
 - 두번째 인자 : 커널의 크기(행,열), 여기서는 3x3의 크기
 - 세번째 인자 : **맨 처음 층**에는 입력되는 값의 크기를 주어야 함(행,열, 색상 - 컬러(3), 흑백(1))
 - 네번째 인자 : 활성화 함수 정의
![1](https://user-images.githubusercontent.com/84856055/121892128-4ba6bf00-cd57-11eb-9d14-4fc0baa2e93f.jpg)
![cnn 커널](https://user-images.githubusercontent.com/84856055/121891865-f66aad80-cd56-11eb-8ba9-f0d41b9eb24b.JPG)
<br><br>
### 패딩(Padding)
 - 컨볼루션 레이어에서 커널을 적용하여 만든 **Feature Map의 크기는 입력 데이터 보다 작음**(ex : (5,5) -> (3,3))
 - Filter를 적용 후 결과 값이 작아지게 되면 처음에 비해서 특징이 유실될 가능성이 큼
 - 이를 방지하기 위해 **입력값 주위로 0 값을 넣어서 입력 값의 크기를 인위적으로 키워서, 결과 값이 작아지는 것을 방지**
 - ***주로 zero Padding(same Padding)을 사용***
 <img width="332" alt="1" src="https://user-images.githubusercontent.com/84856055/121892746-19499180-cd58-11eb-98c7-2a5cc36129b5.png">
 
 ![2](https://user-images.githubusercontent.com/84856055/121893779-55312680-cd59-11eb-8613-7369158c71e3.JPG)
<br><br>
### 풀링(Sub sampling or Pooling)
 - 추출된 Feature Map을 인위로 줄이는 작업을 말함
 - 컨볼루션 레이어를 거쳐서 추출된 특징들은 필요에 따라 풀링 과정이 필요
 - 예를 들어, 고해상도 사진을 보고 물체를 판별할 수 있지만, 작은 사진을 가지고도 물체를 판별할 수 있어야 함
 - 정해진 구역 안에서 최댓값을 뽑아내는 맥스 풀링, 평균값을 뽑아내는 평균 풀링 등이 존재
 - ***주로 맥스 풀링을 사용***
 - 장점
   - 전체 데이터의 크기가 줄어들기 때문에 연산에 들어가는 컴퓨팅 리소스가 적어짐
   - 데이터의 크기를 줄이면서 손실이 발생하기 때문에, 오버피팅을 방지
 ```python3
 model.add(MaxPooling2D(pool_size = 2)) # 풀링 커널의 크기를 2로 정하면 전체 크기가 절반으로 줄어듬
 ```
![3](https://user-images.githubusercontent.com/84856055/121895084-d0470c80-cd5a-11eb-9c07-21850694cae7.JPG)
![4](https://user-images.githubusercontent.com/84856055/121895097-d3da9380-cd5a-11eb-8bad-9494accd15f6.JPG)
<br><br>
### Dropout
- 오버피팅을 막기 위한 방법으로 뉴럴 네트워크가 학습중일때, 은닉층에 배치된 노드를 랜덤하게 꺼서 학습을 방해함으로써, 학습이 학습용 데이타에 치우치는 현상을 막아줌
```python3
model.add(Dropout(0.25)) # 25%의 노드를 끈다
```
![1](https://user-images.githubusercontent.com/84856055/121895915-b6f29000-cd5b-11eb-9035-a5901aa89fb7.jpg)
![1](https://user-images.githubusercontent.com/84856055/121896172-f6b97780-cd5b-11eb-83c6-27eccb9eac30.jpg)
<br><br>
### Flatten
 - 2차원 배열을 1차원으로 바꾸어 주는 함수
 - **컨볼루션 레이어나 맥스 풀링은 주어진 이미지를 2차원 배열인 채로 다룸**
 - 이를 **1차원 배열로 바꿔주어야 활성화 함수가 있는 층에서 사용이 가능**
```python3
model.add(Flatten())
```
![flatting](https://user-images.githubusercontent.com/84856055/121896570-67f92a80-cd5c-11eb-9234-9649cbce257b.JPG)
<br><br> 
### 코드 예시
```python3
model = Sequential()

model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
```
