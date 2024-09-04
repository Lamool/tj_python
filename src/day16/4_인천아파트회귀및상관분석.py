# [1] 가설 : 아파트 층과 건축년도 증가하면서 아파트 거래금액도 비싸다.
# [2] 주제 : 아파트 층과 건축년도에 따른 거래 금액 추이 비교 분석
# [3] 분석 방법 : 다중 회귀분석, 상관분석
# 1. 데이터 수집
import pandas as pd
data = pd.read_csv('아파트(매매)_실거래가_20240904134550.csv', encoding='cp949', skiprows=15, thousands=',')
    # thousands=',' : 천단위 쉼표 생략 # 천단위를 정수타입으로 가져온다.

# 과제 : 해당 csv 파일을 분석하여 제출 (한글 깨짐 무관)

# 독립변수가 층
# 종속변수 거래금액
# grid

# 회귀분석 (다중 선형 회귀분석)
# [1] 모듈 호출
# from statsmodels.formula.api import ols
#
# 회귀모형수식 = '거래금액 ~ 층 + 건축년도'

# 선형회귀모델 = ols(회귀모형수식, data = )



# 상관 분석

# 결측값(누락된 값/공백) 확인
# print(data.isnull().sum())

# 연속형 데이터만 가능하므로 연속형 데이터 열만 추출
data2 = data.select_dtypes(include=[int])


