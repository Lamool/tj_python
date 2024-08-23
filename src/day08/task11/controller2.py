# HTTP 매핑 정의 # 삼성전자 csv 파일 읽어오는 함수
# 1. Flask 객체 필요 , 다른 파일 존재 , 모듈 호출
from app2 import app2         # from 파일명.py import 변수 또는 함수 또는 클래스 또는 *
# 2. Flask 객체를 이용한 HTTP 매핑 정의 # [GET] http://localhost:5000/samsung

@app2.route("/samsung", methods = ["GET"])
def samsungData() :
    stockPriceList = []

    # 1. 파일 읽기 모드
    f = open("삼성전자주가.csv", 'r')

    # 2. 파일 전체 읽어오기
    data = f.read()

    # 3. 데이터 가공 (csv 형식), 행마다 분리
    rows = data.split("\n")

    # 4. 행마다 반복문, 첫 줄 제외
    for row in rows[1:] :
        # 5. 열마다 분리
        cols = row.split(',')

        # 6. 해당 열들을 딕셔너리로 변환
        dic = {     # 일자,종가,대비,등락률,시가,고가,저가,거래량,거래대금,시가총액,상장주식수
            'date': cols[0],
            'closingPrice': format( int( eval( cols[1] ) ), ',' ),      # 천단위 쉼표 : format( 숫자데이터, ',' )
            'contrast': cols[2],
            'fluctuationRate': cols[3],
            'marketPrice': cols[4],
            'highPrice': cols[5],
            'lowPrice': cols[6],
            'tradingVolume': cols[7],
            'transactionAmount': cols[8],
            'marketCapitalization': cols[9],
            'listedStocksNum': cols[10]
        }

        # 7. 리스트 담기
        stockPriceList.append(dic)

    return stockPriceList       # 리스트 반환

# print(samsungData())  # 확인