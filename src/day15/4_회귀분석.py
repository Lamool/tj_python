# [1] 가설 : 출석률이 좋으면 국어점수가 오른다.
# [2] 주제 : 출석률에 따른 국어점수의 비교 분석
# [3] 분석방법 : 출석률(원인/독립변수/연속형) ~ 국어점수(결과/종속변수/연속성)
    # 1. 데이터 수집
import pandas as pd
data = pd.DataFrame({'출석률' : [80, 85, 90, 95, 70],
                     '국어점수' : [60, 65, 75, 85, 50]})
print(data);    # {열이름 : [리스트], 열이름 : [리스트]}
    # 2. 회귀 모형 수식/공식 정의 # 종속변수 ~ 독립변수
Rformula = '국어점수 ~ 출석률'
    # 3. 모델 피팅(해당 모형 수식을 모델에 적용해서 실행)
from statsmodels.formula.api import ols
model = ols(Rformula, data = data).fit
    # 4. 결과

print(model.summary())      # 회귀분석 결과 요약 표 출력
# print(model.params)     # 회귀 계수 출력
# print()
# print(model.tvalues)    # t통계량 출력
#
# print(model.bse)        # 표준오차

# [4] 결론 및 제언
    # 결론 : 출석률이 증가하면 국어 점수가 오른다
    #
    #((회귀 분석 ))
# [5] 한계점 : 결론에 따른 표본의 문제점과 범위 설정
