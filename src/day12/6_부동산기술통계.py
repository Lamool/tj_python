# day12 > 6_부동산기술통계.py
# [1] 인천광역시 부평구 전월세 1년치 csv 수집 # 부동산 실거래가 : https://rt.molit.go.kr/pt/xls/xls.do?mobileAt=
# [2] CSV 파일을 판다스의 데이터프레임 으로 불러오기
# [3] 데이터 탐색(기술통계)
    # 전체 기술통계 결과를 SPRING index6.html 테이블형식으로 출력  ( HTTP 매핑 임의로 정의 )
# [4] 데이터 모델링( 그룹화 )
    # 전월세 기준으로 그룹해서 전용면적의 기술통계 결과를 SPRING index6.html 테이블형식으로 [3]번 테이블 위에 출력  ( HTTP 매핑 임의로 정의 )

# [5] 추가
    # 1. 부평구의 동 명을 중복없이 출력하시오.
    # 2. 가장 거래수가 많은 단지명 을 1~5 등까지 출력하시오.

# [1] 모듈 가져오기
import pandas as pd
from flask import Flask

# [2] CSV 파일을 판다스의 데이터프레임으로 불러오기
apartment = pd.read_csv('아파트(전월세)_실거래가_20240829163333.csv', encoding='cp949')
# print(apartment)

# [3] 데이터 탐색(기술통계)
    # 1. 데이터프레임의 기존 정보 출력
# print(apartment.info())
    # 2. 기술통계
apartment.columns = apartment.columns.str.replace(' ', '_')         # 열이름에 공백이 있으면 _(밑줄)로 변경
# print(apartment.head())
    # .describe() : 속성(열)마다 개수, 평균, std(표준편차), 최솟값, 백분위수25%, 백분위수50%, 백분위수75%, 최댓값
print(apartment.describe())
apartment2 = apartment.describe()


# print(apartment2.index.values)
# result = apartment2.to_json(force_ascii=False)

# print(result)
# print(apartment2.value_counts().to_json())

# print(apartment2.value_counts())
# print(apartment2.index.values.to_json())


# print(len(apartment2))
# # 리스트를 csv 변환해주는 함수
# # def list_to_csv(result)





# [4] 플라스크
# 1. 플라스크 객체 생성
app = Flask(__name__)

# 4. CORS 허용, 서로 다른 port 간의 데이터 통신 허용, 보안상 문제가 발생할 수 있다
from flask_cors import CORS
CORS(app)  # 모든 HTTP에 대해 CORS 허용


# 3. app.run 코드 위에 HTTP 매핑 주소 정의
@app.route('/apartment', methods=['get'])  # http://localhost:5000/apartment
def getApartment():
    # (3) 서비스로부터 받은 데이터로 HTTP 응답하기
    return result


# 2. 플라스크 웹 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)





'''
min	{"(1.0, 5.0, 0.0, 12.0023, 202308.0, 1.0, 0.0, 1.0, 1978.0, 0.0)":1,
25	"(2985.5, 306.0, 0.0, 52.43, 202312.0, 8.0, 0.0, 5.0, 1992.0, 10.0)":1,
std	"(3446.6367664724985, 209.98303960238164, 36.90615674636438, 18.87065503826991, 41.050654405938154, 8.696924983746829, 33.19234683431808, 9.333803613171685, 14.283909047482597, 23.511686094886596)":1,
mean	"(5970.0, 455.575508836586, 9.79512521986766, 63.41527025714046, 202379.6001340146, 15.839014992880475, 31.089538487310495, 12.221877879219365, 2006.642348605411, 35.90449688334817)":1,
50	"(5970.0, 497.0, 0.0, 59.8167, 202403.0, 16.0, 35.0, 10.0, 2005.0, 40.0)":1,
75	"(8954.5, 630.0, 4.0, 69.94, 202405.0, 23.0, 54.0, 17.0, 2022.0, 53.0)":1,
max	"(11939.0, 1002.0, 421.0, 171.67, 202408.0, 31.0, 200.0, 48.0, 2023.0, 168.0)":1,
count	"(11939.0, 11939.0, 11939.0, 11939.0, 11939.0, 11939.0, 11939.0, 11939.0, 11939.0, 4492.0)":1}
'''