# [1] 데이터 수집 # 데이터셋 찾는 과정
# 1. https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
# 2. [다운로드] UCI HAR Dataset 폴더

# 3. 피처(독립변수) 이름 파일 읽어오기 # txt 파일 읽어오기
import pandas as pd
feature_name_df = pd.read_csv('./UCI_HAR_Dataset/UCI_HAR_Dataset/features.txt', sep='\s+', header=None, names=['index', 'feature_name'], engine='python')
    # 1. sep='\s+' : 공백으로 구분된 형식 파일
    # 2. header=None : 제목이 없는 파일
    # 3. names=[열이름]
    # 4. engine='python' 생략
print(feature_name_df.head())
print(feature_name_df.shape)    # (561, 2)
# 4. 인덱스 제거, 독립변수 이름만
feature_name = feature_name_df.iloc[ : , 1].values.tolist()
    #

print(feature_name)

# 훈련용, 테스트용 파일 읽어오기
# ((열 제목 넣어주고 있는 것. 레이블 따로 데이터 따로 가져오고 있어서 합치는 과정인 것))
X_train = pd.read_csv('./UCI_HAR_Dataset/UCI_HAR_Dataset/train/X_train.txt', delim_whitespace=True, header=None, encoding='latin-1')
X_train.columns = feature_name      # 피처(열) 이름 대입
X_test = pd.read_csv('./UCI_HAR_Dataset/UCI_HAR_Dataset/test/X_test.txt', delim_whitespace=True, header=None, encoding='latin-1')
X_test.columns = feature_name       # 피처(열) 이름 대입
Y_train = pd.read_csv('./UCI_HAR_Dataset/UCI_HAR_Dataset/train/y_train.txt', sep='\s+', header=None, names=['action'], engine='python')
Y_test = pd.read_csv('./UCI_HAR_Dataset/UCI_HAR_Dataset/test/y_test.txt', sep='\s+', header=None, names=['action'], engine='python')

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)     # (7352, 561) (7352, 1) (2947, 561) (2947, 1)

print(X_train.head())
print(Y_train['action'].value_counts())

# 종속변수의 데이터 레이블 파일 가져오기
label_name_df = pd.read_csv('./UCI_HAR_Dataset/UCI_HAR_Dataset/activity_labels.txt', sep='\s+', header=None, names=['index', 'label'], engine='python')

# index 제거하고 클래스(종속변수) 분류 값 롤 리스트 추출
label_name = label_name_df.iloc[:, 1].values.tolist()
print(label_name)

# 데이터 수집 정리
'''
    1. activity_labels.txt : 클래스(종족변수) 값에 따른 분류 값
    2. features.txt : 
    
    
    변수
    1. X_train : 독립변수 데이터프레임 (훈련용)
    2. Y_train : 종속변수 데이터프레임
    3. X_test : 독립변수 데이터프레임 (테스트용)
    4. Y_test : 종속변수 데이터프레임
    5. label_name : 종속변수 값에 따른 분류 값    
'''

print(X_train.shape)    # (7352, 561) 행 : 7352, 열 : 561

# 8. 결정트리 모델 구축하기
from sklearn.tree import DecisionTreeClassifier     # 모듈 호출
model = DecisionTreeClassifier()    # 결정 트리 분류 분석 객체 생성
model.fit(X_train, Y_train)     # 피팅 (학습)

# 9. 모델 예측 (샘플 또는 테스트용 데이터)
Y_predict = model.predict(X_test)     # 피팅된 모델이 새로운 데이터의 독립변수를 가지고 종속변수를 예측한다
print(Y_predict)    # [5 5 5 ... 2 1 1]

# 10. 테스트 데이터를 이용한  모델 예측 정확도 확인
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_predict)    # 정확도 확인 # 실제값(Y_test), 예측값(Y_predict)
print(accuracy)     # 0.8517136070580251 # 1에 가까울 수록 예측을 잘 하고 있다


'''
# 11. 최적의 하이퍼 매개변수 찾기, # 최적의 정확도가 높은 트리 찾기 # 정확도 가장 높았을 때의 매개변수를 찾아보자
# 결정트리 모델이 사용하는
# ((아무것도 설정 안 하면 기본값? 진위))

# (1) 결정 트리가 사용하는 하이퍼 매개변수 종류
print(model.get_params())
    # depth : 트리의 깊이 # max_depth : 최대 트리의 깊이
    # criterion : 노드 결정 방식
# (2) 최적의 하이퍼 매개변수를 찾을 설정값을 변수 만들기
params = {
    'max_depth' : [6, 8, 10, 12, 16, 20, 24]        # 다양한 트리의 최대 노드깊이를 설정
}

# (3) 다양한 하이퍼파라미터 조합을 시도해서 최적의 하이퍼파라미터를 찾는데 사용되는 모듈, 교차 검증 제공
from sklearn.model_selection import GridSearchCV
    # cv 객체 생성
    # 미리 설정한 'params'의 'max_depth'라는 최대 노드깊이를 (5회) 교차 검증하는 cv 객체
grid_cv = GridSearchCV(model, param_grid = params, scoring='accuracy', cv = 5, return_train_score=True)
    #

    # cv 객체 테스트
grid_cv.fit(X_train, Y_train)
    # 결과 확인 # .cv_results_
print(grid_cv.cv_results_ )
    # 검증 결과를 데이터프레임 객체로 변환
cv_results_df = pd.DataFrame(grid_cv.cv_results_)
    # 필요한 열(필드) 확인
print(cv_results_df[['param_max_depth', 'mean_test_score', 'mean_train_score']])
    # 최적의 정확도 # .best_score, 최적의 하이퍼 매개변수 확인 # grid_cv.best_params
print(grid_cv.best_score_ , grid_cv.best_params_)
    # 사용처 : 다음에 모델 만들 때 최적의 하이퍼 파라미터를 적용해서 만들기
        # model = DecisionTreeClassifier(max_depth = 16)
'''

# 12. (모델의 성능 개선) 최적의 하이퍼 파라미터 찾기2
params = {
    'max_depth' : [8, 16, 20],              # 트리의 최대 깊이로 검증하겠다.
    'min_samples_split' : [8, 16, 24]       # 노드를 분할 하기 위해 사용 되는 최소 샘플 수의 값들을 검증 하겠다
}

from sklearn.model_selection import GridSearchCV

grid_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=5, return_train_score=True)
grid_cv.fit(X_train, Y_train)
print(grid_cv.best_score_ , grid_cv.best_params_)       # 평균 정확도 :

# 예시] 개션된 모델 생성
model2 = DecisionTreeClassifier(max_mehod=8, min_samples_split=16)
model2.fit(X_train, Y_train)        # 개선된 모델로 다시 피팅
    # 개선된 모델로 다시 테스트
Y_predict2 = model2.predict(X_test)

# cv_results_df = pd.DataFrame(grid_cv.cv_results_)
# cv_results_df[['param_max_depth', 'param_min_samples_split', 'mean_test_score', 'mean_train_score']]
# print('최고 평균 정확도 : {0:.4f}, 최적의 하이퍼 매개변수 : {1}', format)


# 결정트리 모델 시각화
# import matplotlib.pyplot as plt
# from sklearn import tree     # 결정트리 시각화 모듈
# tree.plot_tree(model, feature_names=feature_name, class_names=label_name)
#     # tree.plot_tree(결정트리모델객체, feature_names=[피처이름들], class_names=[클래스이블들])
# plt.show()




