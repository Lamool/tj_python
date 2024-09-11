# 보스톤 주택 데이터 가져오기 # sklearn 1.2 이후로 제공하지 않음.
# from sklearn.datasets import load_boston
# boston = load_boston()
# print(boston)
################### [1] 데이터 수집, 준비 및 탐색
# 보스톤 주택 데이터 가져오기 # sklearn 1.2 # ((위에 거 실행하면 에러 뜨면서 이거 밑에 거 알려줌))
import pandas as pd
import numpy as np
data_url = "http://lib.stat.cmu.edu/datasets/boston"    # 보스턴 주택 정보가 있는 url
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    # 지정한 url에서 데이터를 데이터프레임으로 가져오기
    # sep="\s+" : 데이터 간의 공백으로 구분된 csv
    # skiprows=22 : 위에서부터 22행까지 생략
    # header=None : 헤더가 없다는 뜻
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
# print(data.shape)     # 주택관련변수들 (독립변수, 피처)
# print(target.shape)   # 주택가격 (종속변수, 타킷)

# 독립변수의 이름
feature_names = ['CRIM', 'ZN', 'INDUS','CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# 독립변수 데이터와 독립변수의 이름으로 데이터프레임 생성
boston_df = pd.DataFrame(data, columns=feature_names)
# print(boston_df.head())

# 4. 데이터프레임의 주택가격 열 추가
boston_df['PRICE'] = target
# print(boston_df.head())

print('보스톤 주택 가격 데이터셋 크기 : ', boston_df.shape)  # (506, 14)  # (행개수, 열개수)

print(boston_df.info())     # 열이름, 열의 데이터수, 데이터타입, 메모리

################# [2] 분석 모델 구축 #################
# 1. 타겟과 피처 분할하기
Y = boston_df['PRICE']  # 종속변수, 타겟
X = boston_df.drop(['PRICE'], axis = 1, inplace =False)     # 독립변수, 피처 # 주택 가격 외 정보

# 2. 훈련용과 평가용 분할하기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=156)
# 훈련용독립변수, 테스트용독립변수, 훈련용종속변수, 테스트용종속변수 = train_test_split(독립변수, 종속변수, test_size=분할비율, random_state=난수생성시드)
# test_size=0.3     # 훈련용 70%, 테스트용 30% 분할

print(x_train.shape)    # (354, 13) 70%
print(x_test.shape)     # (152, 13) 30%

# 3. 선형 회귀 분석 모델 생성
from sklearn.linear_model import LinearRegression
lr = LinearRegression()     # 분석 모델 객체

# 4. 모델 훈련
lr.fit(x_train, y_train)    # 훈련용 데이터를 이용한 모델 훈련
print(lr.intercept_)    # y절편
print(lr.coef_)     # 회귀계수

# 5. 테스트용으로 예측 하기 # 테스트용에 있는 주택 정보를 이용한 주택 가격 예측 하기
y_predict = lr.predict(x_test)
print(y_predict)    # 152개의 가격 에측

# 6. 평가지표 확인하기 (MSE, RMSE, 결정계수, Y절편, 회귀계수)
# y_test    # 동일한 피처 정보를 가진 실제 주택가격
# y_predict # 동일한 피처 정보를 가진 예측한 주택가격
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_predict)     # mse 평가
rmse = np.sqrt(mse)                             # rmse 평가
r2 = r2_score(y_test, y_predict)                # r2 평가
print(f'mse : {mse}, rmse : {rmse}, r2 : {r2}')
print(f'y절편 : {lr.intercept_}, 회귀계수 : {np.round(lr.coef_, 1)}')      # np.round(값, 자릿수) : 해당 자릿수에서 반올림 함수

#((오차는 적은 게 좋고)) / 결정계수(?) 1에 가까울 수록 좋다?

# print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
# print('R^2(Variance score) : {0:.3f}'.format(r2_score(y_test, y_predict)))


################# [3] 결과 시각화 #################
# 회귀 분석 결과를 산점도 + 선형 회귀 그래프로 시각화하기
import matplotlib.pyplot as plt
import seaborn as sns   # 희귀분석 관련 차트 구성
fig, axs = plt.subplots(figsize = (16, 16), ncols = 3, nrows = 5)

plt.show()  # 차트 열기
# ((회귀계수 기울기 독립변수가 증가, 감소하는 x값 1증가 할 때마다))
#

# y졀편 : 독립변수가 0일 대 종속변수의 값
# 회귀계수 : 독립변수 1 증가 할 때마다 종속변수의 증갑 단위 # 기울기
# 신뢰구간 : 좋으면 예측이 안정적이고 관계가 명확하다 해석, 넓다면 예측이 불안정하고 관계가 불명확 하다 해석

# (( 기울기 떨어지고 있는 것. -마이너스)




x_features = ['CRIM', 'ZN', 'INDUS','CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

for i, feature in enumerate(x_features) :
    row = int(i/3)  # 목 # 3칸씩
    col = i % 3     # 나머지
    sns.regplot(x = feature, y = 'PRICE', data = boston_df, ax = axs[row][col])
    # ax

    plt.show()  # 차트 열기

