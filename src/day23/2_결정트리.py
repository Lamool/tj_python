# 데이터셋
# 주제 : 여러 어종의 특성(Weight, Length, Diagonal, Height, Width)들을 바탕으로 어종명(Species) 예측하기
# Species : 어종명, Weight : 무게, Length : 길이, Diagonal : 대각선길이, Height : 높이, Width : 너비
# 어종 데이터셋 : https://raw.githubusercontent.com/rickiepark/hg-mldl/master/fish.csv

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# [1] 어종 데이터셋
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/fish.csv')
print(data.head())     # 확인 # (159, 6)
print(data.shape)     # 확인

# 인덱스 제거, 독립변수 이름만
feature_name = data.iloc[:, 1].values.tolist()
print(feature_name)


# [2] 7:3 비율로 훈련용과 테스트용으로 분리 하기

    # 타켓과 피처 분할하기
data = ['Species']    # 종속변수




    # 훈련용과 평가용 분할하기
x_train, x_test, y_train,y_test = train_test_split(X, Y, test_size=0.3, random_state=156)
    # test_size=0.3     # 훈련용 70%, 테스트용 30% 분할


# [3] 결정트리 모델로 훈련용 데이터 피팅 하기
model = DecisionTreeClassifier()        # 결정 트리 분류 분석 객체 생성
model.fit(x_train, y_train)     # 피팅 (학습)


# [4] 훈련된 모델 기반으로 테스트용 데이터 예측하고 정확도 확인하기
# 출력 예시 ] 개선 전 결정트리모델 정확도 : 0.625


# [5] 최적의 하이퍼 파라미터찾기 # params = { 'max_depth' : [2, 6, 10, 14], 'min_samples_split' : [2, 4, 6, 8] }
# 출력 예시 ] 평균 정확도 : x.xxxxxxx, 최적 하이퍼파라미터 : { 'max_depth' : xx, 'min_samples_split' : x }


# [6] 최적의 하이퍼 파라미터 기반으로 모델 개선 후 테스트용 데이터 예측하고 예측 정확도 확인하기 # 시각화하기
# 출력 예시 ] 개선 후 결정트리모델 정확도 : 0.xxx
# 차트 시각화


