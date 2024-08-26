# 연습문제 : 서울 열린데이터 광장에서 민주주의 서울 자유제안을 크롤링하여 JSON 파일로 저장하시오.
# (상상대로 서울 자유제안 정보)


import json
import urllib.request

인증키 = "516a52754c717765383054574f7273"

# [code 2]
def getRequestUrl(url) :
    요청객체 = urllib.request.Request(url)
    try :
        응답객체 = urllib.request.urlopen(요청객체)
        if 응답객체.getcode() == 200 :
            return 응답객체.read().decode('utf-8')
    except Exception as e :
        return None

# [code 3] 지정한 날짜, 지정한 국가, 구분을 받아서 url 요청하기
def getTourismStateItem(yyyymm, nat_cd, ed_cd) :
    # 1. 출입국관광통계의 기본 URL
    base = 'http://openapi.tour.go.kr/openapi/service/EdrcntTourismStatsService/getEdrcntTourismStatsList'
    # 2. 매개변수 : 인증키, 연월, 국가코드
    parameters = f'?_type=json&serviceKey={인증키}'
    parameters += f'&YM={yyyymm}&NAT_CD={nat_cd}&ED_CD={ed_cd}'
    url = base + parameters     # 3. 합치기
    print(f'>> url : {url}')

    responseDecode = getRequestUrl(url)      # 4. url 요청 후 응답객체 받기
    if responseDecode == None : return None     # 5. 만약에 응답객체가 None이면 None 반환
    else : return json.loads(responseDecode)    # 6. 만약에 응답객체가 존재하면 json형식을 파이썬 객체로 반환 함수

# [code 4]
def getTourismStateService(startNum, endNum) :
    jsonResult = []     # 수집한 데이터를 저장할 리스트 객체
    dataEnd = f'{str(endNum)}{str(12)}'  # 마지막 연도의 마지막 월 (12)
    isDataEnd = 0       # 데이터의 끝 확인하는 변수
    for num in range(startNum, endNum + 1) : # 시작 번호부터 마지막 번호까지 반복 # range(부터, 미만)
        print(f'>>num : {num}')
        for month in range(1, 13) :     # 1 ~ 13(미만), 1~12월까지 반복
            print(f'>>month : {month}')
            if isDataEnd == 1 : break   # 만약에 isDataEnd가 1이면 반복문 종료

            # :0>2 오른쪽정렬 >, 2 자릿수, 0 빈칸이면 0 채움  # 1 -> 01, 10 -> 10, 5 -> 05
            yyyymm = f'{str(year)}{str(month):0>2}'
            print(yyyymm)

            # 지정한 날짜, 지정한 국가, 구분을 전달하여 요청 후 응답
            jsonData = getTourismStateItem(yyyymm, nat_cd, ed_cd)
            if jsonData != None :
                print(jsonData)
                # 만약에 지정한 날짜의 내용물이 없으면
                if jsonData['response']['body']['items'] == '' :
                    isDataEnd = 1   # 지정한 날짜에 내용물이 없으므로 반복문 종료
                    print('>> 데이터 없음')
                    break       # 반복문 종료
                # 아니고 내용물이 있으면
                natName = jsonData['response']['body']['items']['item']['natKorNm']     # 국가명
                natName = natName.replace(' ', '')      # 공백 제거
                num = jsonData['response']['body']['items']['item']['num']     # 관광객 수
                # 딕셔너리 : 국가명, 국가코드, 연도월, 방문자수
                dic = { 'nat_name' : natName, 'nat_c' : nat_cd, 'yyyymm' : yyyymm, 'visit_cnt' : num }
                jsonResult.append(dic)      # 딕셔너리를 리스트에 담기
    # 모든 반복문 종료후
    return jsonResult

# [code 1]
def main() :
    # API 서비스 요청시 필요한 매개변수들을 입력받기
    startNum = int(input('시작번호 : '))      # 3. 데이터 수집 시작 번호
    endNum = int(input('끝번호 : '))           # 4. 마지막 데이터의 번호

    # 서비스 요청 후 응답 객체 받기
    jsonResult = []         # 1. 수집한 데이터를 저장할 리스트 객체
    jsonResult = getTourismStateService(startNum, endNum)
    print(jsonResult)   # 확인

    # 7. 응답받은 py 객체를 json으로 변환 후 파일 처리
    with open(f'{startNum}-{endNum}.json', 'w', encoding= "utf-8") as file :
        jsonFile = json.dumps(jsonResult, indent=4, sort_keys=True, ensure_ascii=False)
        file.write(jsonFile)

if __name__ == "__main__" :
    main()



