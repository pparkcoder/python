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
### 단어의 빈도 수 및 index 계산
 - **Tokenizer()함수 이용**
 ```python3
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text=['먼저 텍스트의 각 단어를 나누어 토큰화합니다.',
      '텍스트의 단어로 토큰화해야 딥러닝에서 인식됩니다.',
      '토큰화한 결과는 딥러닝에서 사용할 수 있습니다.',
     ]

t = Tokenizer() # 토큰화 함수 지정
t.fit_on_texts(text) #토큰화 함수에 텍스트 지정
print(t.word_counts) # 각 단어의 빈도 수 출력
print()
print(t.word_index) # 각 단어에 매겨진 index 값 출력
 ```
 ```
 OrderedDict([('먼저',1),('텍스트의',2),('각',1),('단어를',1),('나누어',1),('토큰화',3),('합니다',1),('단어로',1),('해야',1),
              ('딥러닝에서',2),('인식됩니다',1),('한',1),('결과는',1),('사용',1),('할',1),('수',1),('있습니다',1)])
              
 {'딥러닝에서':3, '단어를':6, '결과는':13, '수':16, '한':12, '인식됩니다':11, '합니다':8, '텍스트의':2, '토큰화':1, '할':15,
  '각':5, '있습니다':17, '먼저':4, '나누어':7 ,'해야':10, '사용':14, '단어로':9}
 ```
 <br><br>
### 단어 One - hot encoding
  - 해당 단어가 문장의 어디에서 왔는지, 단어의 순서는 어떤지, 문장의 다른 요소와 어떤 관계를 가지고 있는지 알아보는 방법
  - 각 단어를 모두 0으로 바꾸어 주고 원하는 단어만 1로 바꾸어 주는 것
  - 텍스트 안의 단어들을 숫자의 시퀀스의 형태로 변환하는 **text_to_sequences() 함수 사용**
```python3
encoded=t.texts_to_sequences([text])[0]
print(encoded)
```
```
[2, 5, 1, 6, 3, 7] # 순서는 상관 X
```
  - 0과 1로만 이루어진 배열로 바꾸어주는 **to_categorical()함수 사용**
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
 ### 단어 임베딩(word embedding)
  - One - hot encoding 은 단어의 의미를 전혀 고려하지 않으며 **벡터의 길이가 총 단어 수**가 되므로 길이가 길어진다는 단점
  - 1만 개의 단어 토큰으로 이루어진 문장을 다룰 시, 9999개의 0과 1개의 1로 이루어진 단어 벡터가 생성됨
  - 이러한 공간적 낭비, 단어의 의미를 고려하기 위해 조밀한 차원에 단어를 벡터로 표현한 것
  - 적절한 크기의 배열로 바꾸어 주기 위해 최적의 유사도를 계산하는 **Embegging() 함수 사용**
```python3
from keras.layers import Embegging

model = Sequential()
model.add(Embedding(16,4,input_length = 2)) # 입력될 단어 16 -> 출력되는 벡터 크기 4, 매번 2개씩만 넣겠음
```
<br><br>
### 패딩(Padding) 
 - **딥러닝 모델에 입력을 하려면 학습 데이터의 길이가 동일해야 함**
 - 길이를 똑같이 맞추어 주는 작업
 - 숫자 0을 이용해서 같은 길이의 시퀀스로 변환하는 **pad_sequences() 함수 사용**
```python3
docs = ["너무 재밌네요","최고예요","참 잘 만든 영화예요","추천하고 싶은 영화입니다","한번 더 보고싶네요",
        "글쎄요","별로예요","생각보다 지루하네요","연기가 어색해요","재미없어요"]

# 토큰화 
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
x = token.texts_to_sequences(docs)

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
#### Padding 파라미터
 - post로 지정하면 시퀀스 뒤에 패딩이 채워짐
 - default는 pre
```python3
padded = pad_sequences(sequences, padding='post')

print(padded)
```
```
[[ 5  3  2  4  0  0  0]
[ 5  3  2  7  0  0  0]
[ 6  3  2  4  0  0  0]
[ 8  6  9  2  4 10 11]]
```
#### maxlen 파라미터
 - 시퀀스의 최대 길이를 제한
 - 최대 길이를 넘어가는 시퀀스는 잘라냄
```python3
padded = pad_sequences(sequences, padding='pre', maxlen=6)

print(padded)
```
```
[[ 0  0  5  3  2  4]
[ 0  0  5  3  2  7]
[ 0  0  6  3  2  4]
[ 6  9  2  4 10 11]]
```
#### truncating 파라미터
 - 최대 길이를 넘는 시퀀스를 잘라낼 위치를 지정
 - post로 지정 시 뒷부분을 잘라냄
```python3
padded = pad_sequences(sequences, padding='pre', maxlen=5, truncating='post')

print(padded)
```
```
[[ 0  0  5  3  2]
[ 0  0  5  3  2]
[ 0  0  6  3  2]
[ 8  6  9  2  4]]
```
