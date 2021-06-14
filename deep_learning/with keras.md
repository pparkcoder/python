## keras
파이썬으로 구현된 쉽고 간결한 딥러닝 라이브러리로서 다층퍼셉트론 모델, 컨볼루션 신경망 모델 등 다양한 구성

### 1. 데이터셋 생성하기 + 전처리
 - 원본 데이터를 불러오거나 시뮬레이션을 통해 데이터를 생성하여 훈련셋, 검증셋, 시험셋을 생성
 - 케라스는 데이터를 0에서 1 사이의 값으로 변환한 다음 구동할 때 **최적의 성능**을 보임
 - 전처리 과정에서 0 또는 1로만 이루어진 벡터로 값을 수정하기 위해 **np_utils.to_categorical(클래스, 클래스의 개수) 함수** 사용
```python3
from tensorflow.keras.datasets import mnist 
import tensorflow as ft

(x_train,y_train),(x_test,y_test) = mnist.load_data() #손글씨 data 불러오기

x_train_flat = []    # 전처리
for dat in x_train :
    x_train_flat.append(list(dat.flatten()))
x_train_flat = np.array(x_train_flat) / 255

y_enc = tf.keras.utils.to_categorical(y_train) # One-hot encoding
```
<br>

### 2. 모델 구성
 - Sequential 모델을 생성한 뒤 필요한 레이어를 추가
 - 더 복잡한 모델이 필요한 경우 케라스 함수 API를 이용
```python3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(16,input_dim = 784, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
```
<br>

### 3. 모델 학습과정 설정
 - 학습하기 전, 학습에 대한 설정
 - 손실 함수 및 최적화 방법을 정의
 - compile() 함수 사용
```python3
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])
```
<br>

### 4. 모델 학습
  - 훈련셋을 이용하여 구성한 모델로 학습
  - fit() 함수 사용
```python3
model.fit(x_train_flat,y_enc,epochs = 100, batch_size = 100) #더미코딩 한 y 적용
```
<br>

### 5. 학습과정 점검
  - 모델 학습 시 훈련셋, 검증셋의 손실 및 정확도 측정
  - 반복횟수에 따른 손실 및 정확도 추이를 보며 학습상황 판단
<br>

### 6. 모델 평가
  - 시험셋으로 학습한 모델 평가
  - evaluate() 함수 사용
```python3
model.evaluate(x_train_flat,y_enc)
```
 <br>

### 7. 모델 사용
  - 임의의 입력으로 모델의 출력을 얻음
  - predict() 함수 사용
```python3
xhat = x_test[0:1]  #임의의 데이터 생성
yhat = model.predict(xhat)
print(yhat)
```
