## RNN(Recurrent Neural Network)
- 여러 개의 데이터가 순서대로 입력되었을 때, 앞서 입력받은 데이터를 잠시 기억해 놓는 방법
- 인공 신경망의 한 종류로서, **내부의 순환 구조가 포함**되어 있기 때문에 시간에 의존적이거나 길이가 짧은 순차적인 데이터(Sequential data) 학습에 활용
- 순환이 되는 가운데 앞서 나온 입력에 대한 결과(W.hh)가 뒤에 나오는 입력값에 영향을 줌
![1](https://user-images.githubusercontent.com/84856055/129028098-fa05f7a0-7fdd-4ced-b119-b09ed10b9ee5.JPG)
![1](https://user-images.githubusercontent.com/84856055/129027278-8d64abc5-a6f6-4e0c-9fc7-4b65959aa393.JPG)
- Input 데이터인 x는 Sequence를 가지는 데이터가 되는데 위의 경우는 길이가 6인 데이터이기 때문에 총 6번의 input이 들어감(예를 들면 단어가 6개인 문장)
- RNN 사이에 있는 화살표는 hidden state의 전달을 표현. 최종적으로 y값이 나오는데는 hidden state와 연속적인 x의 입력에 의해 결정 됨
- 예를 들어  어떤 글을 읽을 때, 우리는 한 단어를 읽을 때마다 이전 단어들을 읽으면서 이해했던 내용과 새로운 단어의 이해를 잘 조합하여 새로운 결론을 내림.
- 이때 단어들이 x이고 단어들을 읽으면서 나오는 새로운 결론이 hidden state
<br><br>
### 장기 의존성 문제점 (Long-Term Dependency Problem)
- 순차적인 데이터(Sequential data)의 길이가 길어짐에 따라 신경망을 하나 통과할 때마다 기울기 값이 조금씩 작아져(0으로 수렴) 아무런 정보도 남지 않게 됨
- 반대로 기울기 폭발이 일어나게 되면 오버플로우(overflow)를 일으켜 NaN(Not a Number) 같은 값을 발생시킴
- 즉, **관련 정보와 그 정보를 사용하는 지점 사이 거리가 멀어지는 경우 학습 능력이 현저하게 저하됨**
<br><br>
![1](https://user-images.githubusercontent.com/84856055/129024713-fc1b41ff-b2fe-4b6c-bac2-fe920a274b71.JPG)
<br><br>
## LSTM(Long Short-Term Memory)
- RNN의 단점을 보완한 방법
- 다음 층으로 기억된 값을 넘길지를 관리하는 단계 Cell state를 추가
- 데이터를 계산하는 위치에 입력(Input), 망각(Forget), 출력(Output) 게이트가 추가되어 각 상태 값을 메모리 공간 셀에 저장하고, 데이터를 접하는 게이트 부분을 조정하여 불필요한 연산, 오차 등을 줄여 장기 의존성 문제를 일정 부분 해결
- 직전 데이터뿐만 아니라, ***좀 더 거시적으로 과거 데이터를 고려하여 미래의 데이터를 예측하기 위함***
- 활성화 함수로는 tanh를 사용
![2](https://user-images.githubusercontent.com/84856055/129028353-50cf9238-a3b0-47e4-95dd-20310214aac4.JPG)
![1](https://user-images.githubusercontent.com/84856055/129028520-ddca3da5-e712-4a9f-8644-690ed71b31b9.JPG)
![2](https://user-images.githubusercontent.com/84856055/129028523-8a10e930-9582-415f-81ce-0d17d278d0c7.JPG)
<br><br>
- Hidden state : 이전 출력값
- Input gate : 새로운 정보가 Cell State에 저장 될지 말지를 결정
- Forget gate : 과거의 정보를 버릴지 말지를 결정, sigmoid 활성화 함수를 통해 0 - 1 사이 값 출력
- Output gate: State를 바탕으로 sigmoid 층에 input 데이터를 태워 State의 어느 부분을 출력으로 내보낼지 결정
- [참고링크](https://wegonnamakeit.tistory.com/7)
