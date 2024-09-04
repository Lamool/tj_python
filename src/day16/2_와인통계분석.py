# 목표 : 와인 속성을 분석하여 품질 등급을 예측한다.

# 1. 데이터수집 : winequality-red.csv, winequality-white.csv
# 2. 데이터준비 :
# 2-1. csv 파일의 열 구분자를; 세미콜론 -> ,쉼표 변경하여 cxv 파일을 새로 만들기
import pandas as pd
red_pd = pd.read_csv('winequality-red.csv', sep=';', header=0, engine='python')
    # sep = 'csv구분자'    # 기본값은 ,(쉼표)
    # header : = 0 첫 번째 행을 열 이름으로 지정하겠다는 뜻
white_pd = pd.read_csv('winequality-white.csv', sep=';', header=0, engine='python')
# 새로운 csv 만들기
red_pd.to_csv('winequality-red2.csv', index=False)
    # index=False : 데이터프레임의 인덱스 열은 포함하지 않는다.
white_pd.to_csv('winequality-white2.csv', index=False)

# 2-2. 데이터 병합하기, 레드 와인과 화이트 와인 분석하기 위해 하나로 합치기
print(red_pd.head())        # .head() : 데이터프레임의 위에서부터 5개 행 출력

# 1. 열 추가,  # .insert(삽입할위치, column='열이름', value=값)     0번째(첫번째) 열에 type 열 이름을 red 값들을 추가
red_pd.insert(0, column='type', value='red')
print(red_pd.head())
print(red_pd.shape)     # .shape, (1599, 13)    # 행 개수와 열 개수 반환
print(red_pd.shape[0])

# 2.
print (white_pd.head())
white_pd.insert(0, column='type', value='white')
print(white_pd.head())
print(white_pd.shape)       # (4898, 13)
# 3. 데이터프레임 합치기, pd.concat([데이터프레임1, 데이터프레임2])
wine = pd.concat([red_pd, white_pd])
print(wine.shape)       # (6497, 13)
# 4. 합친 와인 데이터프레임 ---> csv 파일로 저장
wine.to_csv('wine.csv', index=False)

# [3] 데이터 탐색
    # 1. 데이터프레임의 기존 정보 출력
print(wine.info())
    # 2. 기술 통계
wine.columns = wine.columns.str.replace(' ', '_')       # 열이름에 공백이 있으면 _(밑줄)로 변경
print(wine.head())
    # .describe() : 속성(열)마다 개수, 평균, std(표준편차), 최솟값, 백분위수25%, 백분위수50%, 백분위수75%, 최댓값
print(wine.describe())
print(wine.describe()['quality'])       # 와인의 등급 통계
print(wine.describe()['quality'].to_list())   # 와인 등급의 통계 리스트

print(wine['quality'].unique())     # [5 6 7 4 8 3 9]
print(wine.quality.unique())        # [5 6 7 4 8 3 9]   # ((어떤 건['quality'] 어떤 건 quality인 이유가 ['qua lity']- O qua lity - X 특수문자 때문에))
print(sorted(wine.quality.unique()))    # 3 4 5 6 7 8 9     # 와인 등급의 중복 값 제거 하고 정렬

print(wine['quality'].value_counts())      # 특정한 열(등급) 별로 개수
print(wine['quality'].value_counts().to_list())     # [2836, 2138, 1079, 216, 193, 30, 5]
print(wine['quality'].value_counts().to_json())     # {"6":2836,"5":2138,"7":1079,"4":216,"8":193,"3":30,"9":5} ((# "6":2836 - 6등급))

# [4] 데이터 모델링
    # 1. .groupby('그룹기준')['속성명']
    # type 속성으로 그룹해서 quality 속성 기술 통계 구하기
print(wine.groupby('type')['quality'].describe())
    # 2. type속성으로 그룹 해서 quality 속성의 평균
print(wine.groupby('type')['quality'].mean())
    # 3. type 속성으로 그룹 해서 quality 속성의 표준편차
print(wine.groupby('type')['quality'].std())
    # 4. type 속성으로 그룹 해서 quality 속성의 평균, 표준편차.
        # ((mean() 함수와 std() 함수를 묶어서 한 번에 사용하려면 agg() 함수 사용))
print(wine.groupby('type')['quality'].agg(['mean', 'std']))


print('------------------------------')


print(wine.describe()['quality'])
print(wine.describe()['quality'].to_list())   # 와인 등급의 통계 리스트






#############################################
# [1] t-test
    # 원인변수 (독립변수) 레드, 화이트       (명목형)
    # 결과변수 (종속변수) 등급 (1, 2, 3, 4, 5)     (연속형)
# [1] 모듈 호출
from scipy import stats

# [2] 두 집단 표본 만들기
    # 판다스데이터프레임
    # wine(레드와인과 화이트와인 자료를 합친 데이터프레임 객체)
    # .loc(조건, 출력할 열)


#
# print(wine.loc[wine['type'] == 'red', 'quality'])
레드와인등급집단 = wine.loc[wine['type'] == 'red', 'quality']       # type열의 값이 red이면 등급 출력
화이트와인등급집단 = wine.loc[wine['type'] == 'white', 'quality']    # type열의 값이 white이면 등급 출력
# [3] t-test 독립 표본 검증
t통계량, t검증 = stats.ttest_ind(레드와인등급집단, 화이트와인등급집단)
# [4] 결론 확인
print(t통계량)     # -9.685649554187696        # 음수는 첫 번째 집단의 평균이 두 번째 집단보다 낮다는 뜻  # 해석 : 화이트등급이 평균적으로 9.68 차이가 있다.
print(t검증)      # 4.888069044201508e-22     # e-소수부크기 # ((4.8이 아니라 0.000000...0488...?))

if t검증 < 0.05 : print("해당 가설은 유의미 하다")
else : print("해당 가설은 무의미 하다.")



# ----------------------------------------------------------------------------------

# 2. 회귀분석 (다중 선형 회귀분석)
# [1] 모듈 호출
from statsmodels.formula.api import ols     # [statsmodels] 모듈 설치

# [2] 회귀모형수식(종속변수와 독립변수를 구성하는 방식/공식 : 종속변수명 ~ 독립변수1 + 독립변수2 + 독립변수3)
회귀모형수식 = 'quality ~ fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol'        # 종속변수와 독립변수를 정의 ((실제 표본 집단에 있는 걸 넣는 거라 오타 내면 안 됨. ex) alcohol2라 적을 경우 오류))
    # 가설 : 얄코올 수치에 따라 등급 확인
    # 독립변수(원인/연속성) 알콜
    # 종속변수(결과/연속성) 등급

# [3] ols(선형 회귀 모델) # ols(회귀모형수식, data = 표본집단)    # 최소 제곱법
선형회귀모델 = ols(회귀모형수식, data = wine)
선형회귀모델결과 = 선형회귀모델.fit()
# print(선형회귀모델결과.summary())
print(선형회귀모델결과.params)      # 모델 결과 확인 (회귀 계수)
print(선형회귀모델결과.pvalues)     # 모델 결과 확인 (p값)

# 학습된 회귀분석 모델로 새로운 샘플 예측하기
# 1. 기존 와인 정보를 가지고 등급과 타입을 제외한 새로운 데이터프레임
sample1 = wine[wine.columns.difference(['quality', 'type'])]
print(sample1.head())
# 2. 5행 추출
sample1 = sample1[0:5][:]       # 5행, 모든 열 추출
print(sample1)
# 3. 5개 와인 샘플 정보의 등급을 예측하자
와인등급예측결과 = 선형회귀모델결과.predict(sample1)
print("----- 샘플 데이터의 와인 등급 예측 -----")
print(와인등급예측결과)
'''
0    4.997607
1    4.924993
2    5.034663
3    5.680333
4    4.997607
dtype: float64
'''
print("----- 샘플 데이터의 와인 등급 -----")
print(wine[0:5]['quality'])
'''
0    5
1    5
2    5
3    6
4    5
Name: quality, dtype: int64
'''

# 4. 새로운 와인 정보 2개를 입력 (type, quality 제외한)
newData = {"fixed_acidity" : [8.5, 8.1], "volatile_acidity":[0.8, 0.5], "citric_acid":[0.3, 0.4],
           "residual_sugar":[6.1, 5.8], "chlorides":[0.055, 0.04], "free_sulfur_dioxide":[30.0, 31.0],
           "total_sulfur_dioxide":[98.0, 99], "density":[0.996, 0.91], "pH":[3.25, 3.01],
           "sulphates":[0.4, 0.35], "alcohol":[9.0, 0.88]}
sample2 = pd.DataFrame(newData, columns=sample1.columns)
print(sample2)  # (type, quality 제외한)
'''
   alcohol  chlorides  ...  total_sulfur_dioxide  volatile_acidity
0     9.00      0.055  ...                  98.0               0.8
1     0.88      0.040  ...                  99.0               0.5
'''

# 5. 새로운 2개의 와인 정보를 가지고 등급(quality) 예측
와인등급예측2 = 선형회귀모델결과.predict(sample2)
print(와인등급예측2)
'''
0    4.809094
1    7.582129
dtype: float64
'''


# 히스토그램 차트
# 와인 유형에 따른 품질 등급 히스토그램 그리기
import matplotlib.pyplot as plt
import seaborn as sns   # install seaborn
sns.set_style('dark')   # 1. 히스토그램의 차트 배경색 설정
# 2. distplot 객체 생성
sns.distplot(레드와인등급집단, kde = True, color = "red", label = 'red wine')
sns.distplot(화이트와인등급집단, kde = True, label = 'white wine')
    # kde : 커널 밀도 추정
    # 밀도 : 어떤 값 또는 구간에 데이터가 얼마나 집중되어 있는지 나타내는 값
plt.title("Quality of Wine Type")       # 3. 차트 제목
plt.legend()        # 4. 차트 범례 표시
plt.show()          # 5. 차트 보기



# 부분 회귀 플롯으로 시각화하기
import statsmodels.api as sm
# 1. 부분 회귀에 사용할 등급(종속)과 고정산(독립) 제외한 모든 열 이름을 리스트로 구성한다.
others = list(set(wine.columns).difference(set(["quality", "fixed_acidity"])))
    # 1. list() : 리스트 타입 변환 함수
    # 2. set() : 집합 타입 변환 함수 [중복이 없는 컬렉션]
print(others)

sm.graphics.plot_partregress("quality", "fixed_acidity", others, data = wine, ret_coords = True)     # plot_partregress : 부분회귀
    # quality : 종속변수
    # fixed_acidit : 독립변수
    # others : 다중회귀 분석에서 분석할 부분 독립변수를 제외한 나머지 독립 변수 리스트 , 고정산 과 등급의 관계에서 다른 변수들이 미치는 영향을 제거
    # data = win : 분석에 사용되는 데이터프레임

# sm.graphics.plot_partregress_grid(regression_result, fig = fig)
# 3. 차트 표시
plt.show()

# 다중회귀분석 결과를 부분회귀 플랫으로 그리드형식으로 차트
fig = plt.figure(figsize = (8, 13))     # 차트 크기 설정
sm.graphics.plot_partregress_grid(선형회귀모델결과, fig = fig)     # 선형회귀모델 결과를 각 부분별 회귀 플롯을 그리드 형식으로 구성
plt.show()


'''
    - 각 플롯(차트)에서 독립변수(각 와인속성)와 종속변수(등급) 간의 선형 관계의 강도를 차트로 표시
    - 기울기, 잔차의 패턴, 점들의 분포 등을 관찰하여 독립변수가 종속변수에 미치는 영향 확인
    
    ((예측된 가격과 실제 가격의 차이 - 잔차??))
    ((점들이 선형에 가까울 수록 잔차가 0에 가까운 것))
    ((잔차가 크면 예측을 못한 거다))
    - 예) 아파트 가격을 예측하는 모델을 만들었는데, 실제 가격은 10억인데 모델이 예측한 가격은 8억이라고 했을 대 잔차는 2억이다.
        - 즉) 잔차가 0에 가까우면 모델이 데이터 포인트를 잘 예측했다.
        - 잔차가 크면 모델이 데이터 포인트를 잘 예측하지 못했다.
        - 잔차는 모델 성능을 평가하고 계산할 부분을 찾는데 중요한 역할
    - 플롯이 선형적이고 잔차가 무작위 분포 하면 데이터가 잘 설명되고 있는 상태, 점들이 선형에서 크게 벗어나면 모델을 개선할 필요가 있다.
    
        
    
'''

