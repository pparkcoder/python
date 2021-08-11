## RNN(Recurrent Neural Network)
- 여러 개의 데이터가 순서대로 입력되었을 때, 앞서 입력받은 데이터를 잠시 기억해 놓는 방법
- 인공 신경망의 한 종류로서, **내부의 순환 구조가 포함**되어 있기 때문에 시간에 의존적이거나 길이가 짧은 순차적인 데이터(Sequential data) 학습에 활용
- 순환이 되는 가운데 앞서 나온 입력에 대한 결과(W.hh)가 뒤에 나오는 입력값에 영향을 줌
![1](https://user-images.githubusercontent.com/84856055/129024281-f1585d14-546f-4ec3-9d04-7113b8d9847b.JPG)
<br><br>
### 장기 의존성 문제점 (Long-Term Dependency Problem)
- 순차적인 데이터(Sequential data)의 길이가 길어짐에 따라 신경망을 하나 통과할 때마다 기울기 값이 조금씩 작아져(0으로 수렴) 아무런 정보도 남지 않게 됨
- 반대로 기울기 폭발이 일어나게 되면 오버플로우(overflow)를 일으켜 NaN(Not a Number) 같은 값을 발생시킴
- 즉, 관련 정보와 그 정보를 사용하는 지점 사이 거리가 멀어지는 경우 학습 능력이 현저하게 저하됨
![1](https://user-images.githubusercontent.com/84856055/129024713-fc1b41ff-b2fe-4b6c-bac2-fe920a274b71.JPG)

