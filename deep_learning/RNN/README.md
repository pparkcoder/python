## 자연어 처리(Natural Language Processing)
 - 자연어란 우리가 평소에 말하는 음성이나 텍스트
 - 자연어 처리는 이러한 음성이나 텍스트를 컴퓨터가 인식하고 의미를 분석할 수 있도록 하는 것
 - ***컴퓨터는 수치로 된 데이터만 이해하기에 텍스트를 정제하는 전처리 과정이 꼭 필요***
<br><br>
### 텍스트의 토큰화
 - 토큰(Token) : 텍스트를 단어별, 문장별, 형태소별로 나눌 수 있는데, 이렇게 작게 나누어진 하나의 단위
 - 토큰화(Tokenization) : 입력된 데이터를 잘게 나누는 과정
 - **text_to_word_sequence() 함수를 이용**
```python3
from tensorflow.keras.preprocessing.text import text_to_word_sequence

text = "해보지 않으면 해낼 수 없다"
result = text_to_word_sequence(text)
```
```
['해보지', '않으면', '해낼', '수', '없다']
```
<br><br>
#### 딥러닝에 적용하기 위해서는 각 단어에 매겨진 index값을 얻어야 함
 - **Tokenizer()함수 이용**
 ```python3
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text="나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

t = Tokenizer() # 토큰화 함수 지정
t.fit_on_texts([text]) #토큰화 함수에 텍스트 지정
print(t.word_index) # 각 단어에 매겨진 index 값 출력
 ```
 ```
 {'갈래': 1, '점심': 2, '햄버거': 3, '나랑': 4, '먹으러': 5, '메뉴는': 6, '최고야': 7}
 ```
 <br><br>
### 단어 One - hot encoding
  - 각 단어를 모두 0으로 바꾸어 주고 원하는 단어만 1로 바꾸어 주는 것
  - Tokenizer의 **text_to_sequences() 함수 사용**
```python3
encoded=t.texts_to_sequences([text])[0]
print(encoded)
```
```
[2, 5, 1, 6, 3, 7] # 순서는 상관 X
```
  - 0과 1로만 이루진 배열로 바꾸어주는 **to_categorical()함수 사용**
```python3
one_hot = to_categorical(encoded)
print(one_hot)
```
```
[[0. 0. 1. 0. 0. 0. 0. 0.] #인덱스 2의 원-핫 벡터
 [0. 0. 0. 0. 0. 1. 0. 0.] #인덱스 5의 원-핫 벡터
 [0. 1. 0. 0. 0. 0. 0. 0.] #인덱스 1의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 1. 0.] #인덱스 6의 원-핫 벡터
 [0. 0. 0. 1. 0. 0. 0. 0.] #인덱스 3의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 0. 1.]] #인덱스 7의 원-핫 벡터
 ```
 <br><br>
 ### 단어 임베딩(Embedding)
  - One - hot encoding 은 단어의 의미를 전혀 고려하지 않으며 **벡터의 길이가 총 단어 수**가 되므로 길이가 길어진다는 단점
  - 이러한 공간적 낭비, 단어의 의미를 고려하기 위해 조밀한 차원에 단어를 벡터로 표현한 것
  - **Embegging() 함수 사용**
```python3
from keras.layers import Embegging

model = Sequential()
model.add(Embedding(16,4,input_length = 2)) # 입력될 단어 16 -> 출력되는 벡터 크기 4, 매번 2개씩만 넣겠음
```
<br><br>
### 패딩(Padding) 
 - 딥러닝 모델에 입력을 하려면 학습 데이터의 길이가 동일해야 함
 - 길이를 똑같이 맞추어 주는 작업
 - **pad_sequences() 함수 사용**
```python3
docs = ["너무 재밌네요","최고예요","참 잘 만든 영화예요","추천하고 싶은 영화입니다","한번 더 보고싶네요","글쎄요","별로예요","생각보다 지루하네요","연기가 어색해요","재미없어요"]

# 토큰화 
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
x = token.texts_to_sequences(docs)
print("\n리뷰 텍스트, 토큰화 결과:\n",  x)

padded_x = pad_sequences(x, 4)  # 서로 다른 길이의 데이터를 4로 맞추기
print("\n패딩 결과:\n", padded_x)
```
```
 [[ 0  0  1  2]
 [ 0  0  0  3]
 [ 4  5  6  7]
 [ 0  8  9 10]
 [ 0 11 12 13]
 [ 0  0  0 14]
 [ 0  0  0 15]
 [ 0  0 16 17]
 [ 0  0 18 19]
 [ 0  0  0 20]]
```
