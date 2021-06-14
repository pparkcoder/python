## CNN(Convolutional Neural Network)
 - 컨볼루션 신경망으로서, 입력된 이미지에서 다시 한번 특징을 추출하기 위해 커널(슬라이딩 윈도)을 도입하는 기법
 - 추출된 특징을 기반으로 기존의 뉴럴 네트워크를 이용하여 분류

### 합성곱(Convolution)
두 함수 f,g 가운데 하나의 함수를 반전(reverse), 전이(shift) 시킨 다음, 다른 하나의 함수와 곱한 결과를 적분
<br><br>
### 필터, 커널 (Filter = Kernel)
- 이미지의 특성을 찾아내기 위한 공용 파라미터로, 어떤 특징이 데이터에 있는지 없는지를 검출하는 역할
- 일반적으로 (4,4) 또는 (3,3)과 같은 정사각 행렬로 정의
- 입력 데이터를 지정된 간격으로 순회하며 합성곱을 통해서 만든 출력물 Feature Map을 만듬
![1](https://user-images.githubusercontent.com/84856055/121892128-4ba6bf00-cd57-11eb-9d14-4fc0baa2e93f.jpg)
![cnn 커널](https://user-images.githubusercontent.com/84856055/121891865-f66aad80-cd56-11eb-8ba9-f0d41b9eb24b.JPG)
#### Conv2D()
```python3
model.add(Con2D(32, kernel_size = (3,3), input_shape = (28,28,1), activation = 'relu'))
```
 - 컨볼루션 층(합성곱 층)을 추가하는 함수
 - 첫번째 인자 : 적용할 커널의 수, 여기서는 32개
 - 두번째 인자 : 커널의 크기(행,열), 여기서는 3x3의 크기
 - 세번째 인자 : **맨 처음 층**에는 입력되는 값의 크기를 주어야 함(행,열, 색상 - 컬러(3), 흑백(1))
 - 네번째 인자 : 활성화 함수 정의
<br><br>
### 패딩(Padding)
 - 컨볼루션 레이어에서 커널을 적용하여 만든 **Feature Map의 크기는 입력 데이터 보다 작음**(ex : (5,5) -> (3,3)
 - Filter를 적용 후 결과 값이 작아지게 되면 처음에 비해서 특징이 유실될 가능성이 큼
 - 이를 방지하기 위해 **입력값 주위로 0 값을 넣어서 입력 값의 크기를 인위적으로 키워서, 결과 값이 작아지는 것을 방지**
 - 주로 zero Padding을 사용
 <img width="332" alt="1" src="https://user-images.githubusercontent.com/84856055/121892746-19499180-cd58-11eb-98c7-2a5cc36129b5.png">
 
 ![2](https://user-images.githubusercontent.com/84856055/121893779-55312680-cd59-11eb-8613-7369158c71e3.JPG)
