'''
    1. 분류분석
        1. 로지스틱 회귀분석 : 주로 이진 분류
        2. 결정트리 분석 : 주로 다중 분류 , 여러 개의 클래스로 분류
        - 피처 , 독립변수
        - 클래스, 타겟, 종속변수
    2. 결정트리란?
        - 트리 구조 기반으로 의사 결정 해서 조건을 규칙노드로 나타내고 최종적인 리프노드로 결과를 제공
            - 루트 노드 : 트리의 최상단 위치 하는 노드,
            - 내부/규칙 노드 : 속석(특징)에 기반해 데이터를 분할하는 기준 되는 모드
            3. 리프 노드 : 더 이상 분할되지 않고 최종적인 결과 노드
        - 노드 선택 기준
            1. 엔트로피 : 정보이득지수

            2. 지니 계수
            피처
            지니 계수


'''
# [1] 데이터 샘플
data = {
    'size': [1, 2, 3, 3, 2, 1, 3, 1, 2, 3, 2, 1, 3, 1, 2],  # 과일의 크기
    'color': [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 2, 2, 3, 3],  # 1: 빨간색, 2: 주황색, 3: 노란색
    'labels': [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 1, 0, 2, 2]  # 0: 사과, 1: 오렌지, 2: 바나나
}

# [2] 데이터 프레임 생성
import pandas as pd
df = pd.DataFrame(data)
print(df)

# [3] 독립변수/피처, 종속변수/클래스/타겟 나누기
x = df[ ['size', 'color'] ]
print(x)

y = df['labels']
print(y)

# [4] 결정 트리 모델 생성
from sklearn.tree import DecisionTreeClassifier # 모델 모듈 호출
model = DecisionTreeClassifier()

# [8] 훈련용, 테스트용 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# [5] 모델 피팅
model.fit(x_train, y_train)

# [8] 예측
y_pred = model.predict(x_test)

# [9] 정확도
from sklearn.metrics import accuracy_score # 정확도 함수
accuracy = accuracy_score(y_test, y_pred)   # accuracy_score(실제값, 예측값) 정확도 확인
print(accuracy)


# [6] 확인
print(model.get_depth())    # 트리의 깊이 # 2
print(model.get_n_leaves())     # 리프 노드의 개수 3

# [7] 시각화
import matplotlib.pyplot as plt
from sklearn import tree
tree.plot_tree(model, feature_names=['size', 'color'], class_names=['apple', 'orange', 'banana'])
plt.show()


