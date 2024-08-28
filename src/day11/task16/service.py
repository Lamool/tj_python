# 1. 모듈 가져오기
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import json

# [1] 잡코리아 정보 크롤링 서비스
def jobkoreaInfo(result) :
    for page in range(1, 2) :
        # 2. 지정한 url을 호출해서 응답 받기
        url = f"https://www.jobkorea.co.kr/Search/?stext=java&tabType=recruit&Page_No={page}"
        response = urllib.request.urlopen(url)
        if response.getcode() == 200 :
            print(">>통신 성공")
            # 3. 통신 응답 결과를 읽어 와서 크롤링 파싱 준비
            soup = BeautifulSoup(response.read(), "html.parser");  # print(soup)
            # 4. 분석한 HTML 식별자들을 파싱, find, findAll, select, select_one
            # 4-1. 리스트 전체 파싱
            list = soup.select_one('.list');     # print(list)      # 4-1. 리스트 전체 파싱
            rows = list.select('.list-item');    # print(rows)         # 4-2. 리스트(전체목록) 마다 행(매장) 파싱
            # num = 1     # 확인. 지우기
            for row in rows:  # 4-3. 행(매장) 마다
                #print(num)      # 확인. 지우기
                # print('---------------------------------------------------------------')    # 확인. 지우기
                #print(row)
                cols = row.select('div');       # print(cols)   # 4-3 열 (각 정보) 파싱
                # print('~~~~~~~~~~~~~~~~')  # 확인. 지우기
                # 각 정보들을 파싱
                회사명 = cols[0].select('a')[0].string.strip();         # print(회사명)
                # 공고명 = cols[1].select_one('div').select('a')[0].text.strip();        # print(공고명)
                공고명 = cols[1].select_one('.information-title > a').text.strip();    # print(공고명)
                경력 = cols[1].select_one('.chip-information-group').select('li')[0].text.strip();          # print(경력)
                학력 = cols[1].select_one('.chip-information-group').select('li')[1].text.strip();          # print(학력)
                계약유형 = cols[1].select_one('.chip-information-group').select('li')[2].text.strip();       # print(계약유형)
                지역 = cols[1].select_one('.chip-information-group').select('li')[3].text.strip();          # print(지역)

                count = len(cols[1].select_one('.chip-information-group').select('li'))         # li 개수 구하기
                # print(len(cols[1].select_one('.chip-information-group').select('li')))        # li 개수 구하기

                if count >= 5 :
                    채용기간 = cols[1].select_one('.chip-information-group').select('li')[4].text.strip();       # print(채용기간)
                else :
                    채용기간 = ''

                # num += 1        # 확인. 지우기

                일자리정보 = [회사명, 공고명, 경력, 학력, 계약유형, 지역, 채용기간];     # print(일자리정보) # 리스트에 담기
                result.append(일자리정보)         # 리스트에 파싱한 리스트 담기   # 2차원 리스트

        else :
            print(">>통신 실패")

    # 7. 리스트 반환
    return result

# [2] 2차원 리스트를 csv변환해주는 서비스
def list2d_to_csv(result, fileName, colsNames) :
    try :
        # 1. import pandas as pd 모듈 호출
        # 2. 데이터프레임 객체 생성, 데이터 ,열 목록
        df = pd.DataFrame(result, columns=colsNames)
        # 3. 데이터프레임 객체를 CSV 파일로 생성하기
        df.to_csv(f'{fileName}.csv', encoding='utf-8', mode='w')
        return True
    except Exception as e :
        print(e)
        return False

# [3] csv 파일을 JSON 형식의 PY타입으로 가져오기, 매개변수 - 가져올파일명
def read_csv_to_json(fileName) :
    # 1. 판다스를 이용한 csv를 데이터프레임으로 가져오기
    df = pd.read_csv(f'{fileName}.csv', encoding='utf-8', engine='python', index_col=0)
        # index_col=0 : 판다스의 데이터프레임워크 형식 유지 (테이블형식) ((-> 맨위 왼쪽칸 비어 있는데 이거 유지하도록 하는 거??))
    # 2. 데이터프레임 객체를 JSON으로 가져오기
    jsonResult = df.to_json(orient='records', force_ascii=False)
        # to_json() : 데이터프레임 객체 내 데이터를 JSON 변환함수
            # oreint='records' : 각 행마다 하나의 JSON 객체로 구성
            # force_ascii=False : 아스키 문자 사용 여부 : True(아스키), False(아스키 대신 유니코드 utf-8)
    # 3. JSON 형식 (문자열 타입)의 py타입(객체타입-리스트/딕셔너리)으로 변환
    result = json.loads(jsonResult)
    return result

#  [4] 총채용공고수, 경력별 공고수 ,학력별 공고수
def announcement_number(result) :
    totalEmployment = len(result)         # print(totalEmployment)    # 총채용공고수





# 서비스 테스트 확인 구역
# if __name__ == "__main__" :
#     result = []
#     jobkoreaInfo(result);       print(result);      # 잡코리아 정보 크롤링 서비스 호출
#     list2d_to_csv(result, '잡코리아일자리정보', ['회사명', '공고명', '경력', '학력', '계약유형', '지역', '채용기간'])        # 잡코리아 정보를 csv로 저장 서비스 호출
#     result2 = read_csv_to_json('잡코리아일자리정보')         # csv 파일을 json으로 가져오는 서비스 호출
#     print(result2)

