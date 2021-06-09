## overfitting (과적합)
- 모델이 학습 데이터셋 안에서는 일정 수준 이상의 예측 정확도를 보이지만, **새로운 데이터에 적용하면 잘 맞지 않는 것**
- 과적합은 층이 너무 많거나, 변수가 복잡해서 발생하기도 하고 **테스트셋과 학습셋이 중복될 때 생기기도 함**
- 층을 더하거나 epoch 값을 높여 실행 횟수를 늘리면 정확도가 올라갈 수 있다. 하지만 학습 데이터셋만으로 평가한 예측 성공률이 테스트셋에 그대로 적용x
- 학습이 깊어져서 학습셋 내부에서의 성공률은 높아져도 테스트셋에서는 효과가 없다면 과적합이 일어나고 있는 것<br><br>
![overfitting](https://user-images.githubusercontent.com/84856055/121381437-aff50780-c980-11eb-9c82-cfa23f8bffbb.JPG)
<br>

## train_test_split()
- 불러온 X 데이터와 Y 데이터에서 각각 정해진 비율(%) 만큼 구분하여 **한 그룹은 학습에 사용, 다른 한 그룹은 테스트에 사용하는 함수**
- sklearn 라이브러리에 포함
- **데이터가 충분하지 않다면 좋은 결과값을 얻기 어렵다는 단점**
```python3
from sklearn.model.selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = seed) # 학습셋을 70%, 테스트셋을 30%

model.fit(x_train, y_train, epochs = 130, batch_size = 5)
print("Accuracy : %.4f" % (model.evaluate(x_test, y_test)[1]))
```
   
## K-Cross-Validation (K겹 교차 검증)
 - 데이터가 충분하지 않다면 좋은 결과값을 얻기 어려운 train_test_split()을 보완하는 방법
 - **데이터 셋을 여러 개로 나누어 하니씩 테스트셋으로 사용하고 나머지를 모두 합해서 학습셋으로 사용하는 방법**
 - sklearn의 StratifielKFold() 함수 사용<br>
```python3
from sklearn.model_selection import StratifielKFold

skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
.... # 생략
```
![image](https://user-images.githubusercontent.com/84856055/121383467-71604c80-c982-11eb-9358-9011b4085786.png)
<br><br>

# 모델 저장과 재사용 관련
학습이 끝난 후 테스트해 본 모델을 저장하여 새로운 데이터에 사용할 수 있다.
```python3
from keras.models import load_model

model.save('모델 명.h5') # 모델 저장

model = load_model('사용할 모델 명.h5') # 모델 불러오기
print("Test Accuracy : %.4f % (model.evaluate(x_test, y_test)[1])) # 불러온 모델로 테스트 실행
```

