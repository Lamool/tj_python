# task12->> controller.py

from app import app

@app.route('/apartment', methods=["GET"])
def incheonApartment():
    apartmentList = []
    #파일 읽기
    f=open('아파트(매매)_실거래가_20240823154222.csv', 'r')
    data=f.read()
    rows=data.split("\n")
    for row in rows[16: len(rows)-1]:
        cols=row.split(',')
        print( cols )
        if len( cols ) == 21 :
            dic={
                'city_county_district' : cols[1] ,  #시군구
                'complex_name':cols[5] ,            #단지명
                'exclusive_area':cols[6],           #전용면적
                'contract_year_month':cols[7] ,     #계약연월
                'contract_date':cols[8] ,           #계약일
                'transaction_amount':cols[9]+cols[10],       #거래금액
                'floor':cols[12]                     #층
                 }

            apartmentList.append(dic)
    f.close()
    return apartmentList


@app.route('/apartment2', methods=["GET"])
def incheonApartment2():
    apartmentList = []
    # 파일 읽기
    f = open('아파트(매매)_실거래가_20240823154222.csv', 'r')
    data = f.read()
    rows = data.split("\n")
    for row in rows[16:27584]:
        cols = row.split(',')
        print( cols )
        if len(cols) == 21:
            dic = {
                'city_county_district': cols[1],  # 시군구
                'complex_name': cols[5],  # 단지명
                'exclusive_area': cols[6],  # 전용면적
                'contract_year_month': cols[7],  # 계약연월
                'contract_date': cols[8],  # 계약일
                'transaction_amount': int( cols[9].replace('"','') +  cols[10].replace('"','') ),  # 거래금액
                'floor': cols[12]  # 층
            }
            apartmentList.append(dic)

    #
    maxDic = apartmentList[0] # 거래대금 최댓값이 저장된 딕셔너리 인덱스
    minDic = apartmentList[0] # 거래대금 최솟값이 저장된 딕셔너리 인덱스

    for dic in apartmentList :
        if maxDic['transaction_amount'] < dic['transaction_amount'] :
            maxDic = dic
    for dic in apartmentList :
        if minDic['transaction_amount'] > dic['transaction_amount'] :
            minDic = dic

    print( maxDic )
    print(minDic)

    return apartmentList

#[조건3] OO 구별 거래량 수 계산해서 출력
@app.route('/apartment3', methods=["GET"])
def incheonApartment3():
    apartmentList = ["","인천광역시 부평구 용현동"]
    # 파일 읽기
    f = open('아파트(매매)_실거래가_20240823154222.csv', 'r')
    data = f.read()
    rows = data.split("\n")
    for row in rows[16:27584]:
        cols = row.split(',')
        print( cols )
        if len(cols) == 21:
            dic = {

                'city_county_district': cols[1] ,  # 시군구
                'complex_name': cols[5],  # 단지명
                'exclusive_area': cols[6],  # 전용면적
                'contract_year_month': cols[7],  # 계약연월
                'contract_date': cols[8],  # 계약일
                'transaction_amount': int( cols[9].replace('"','') +  cols[10].replace('"','') ),  # 거래금액
                'floor': cols[12]  # 층
            }
            apartmentList.append(dic)

            city_county_district_parts = dic['city_county_district'].split()
            if len(city_county_district_parts) >= 1:
                dic['city_county_district'] = city_county_district_parts[1]  # '미추홀구'를 선택

            print(dic)

    #
    return apartmentList


