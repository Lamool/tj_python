# http://www.qooqoo.co.kr/bbs/board.php?bo_table=storeship
# 1. BeautifulSoup 이용한 쿠우쿠우 전국 매장 정보 크롤링
# 2. 전국 쿠우쿠우 매장 정보(번호, 매장명, 연락처, 주소, 영업시간)
# 3. pandas 이용한 csv 파일로 변환
# 4. 플라스크 이용한 쿠우쿠우 전국 매장 정보 반환하는 HTTP 매핑 정의한다
    # HTTP(GET) ip주소:5000/qooqoo
    # (3) 생성된 csv 파일 읽어서(pandas DataFrame) json 형식을 반환

from bs4 import BeautifulSoup
import urllib.request


# [code 1]
def qooqoo_store(result) :
    for page in range(1, 2) :    # 1부터 6까지 반복   # 1, 7로 바꿔야 함. 테스트 하느라 1, 2
        # 쿠우쿠우 매장 정보 url
        url = f"http://www.qooqoo.co.kr/bbs/board.php?bo_table=storeship&&page={page}"
        response = urllib.request.urlopen(url)
        soup = BeautifulSoup(response, "html.parser");    # print(soup)
        tbody = soup.select_one('tbody');         # print(tbody)
        num = 0

        for row in tbody.select('tr') :
            # print(row)
            tds = row.select('td')

            #if num % 2 == 0 :  num이 짝수일 때만 result에 넣도록 하려 했는데 이렇게 하면 안 된다고
                # print(tds)
                # print('----------------------------------')
                # print(tds[1])
                # print('----------------------------------')
                # print(tds[1].find('div', attrs={'class' : 'td-subject ellipsis'}).text)
                # print('----------------------------------')
                # print(tds[1].findAll('a')[1].string)
            if len(tds) <= 1: continue
            print(tds)
            print('----------------------------------')
            print(tds[1])
            print('----------------------------------')
            print(tds[2].findAll('a')[0].string)  # 공백제거 해야 함 strip() 함수
            store_num = tds[0].string;
            print(tds[0].string)  # 번호
            store_name = tds[1].findAll('a')[0].string.strip() + tds[1].findAll('a')[1].string.strip();
            print(store_name)  # 매장명
            store_phone = tds[2].findAll('a')[0].string;
            print(tds[2].findAll('a')[0].string)  # 연락처
            store_address = tds[3].findAll('a')[0].string;
            print(tds[3].findAll('a')[0].string)  # 주소
            store_time = tds[4].findAll('a')[0].string;
            print(tds[4].findAll('a')[0].string)  # 영업시간
            # num += 1




# [code 0]
def main() :
    result = []     # 쿠우쿠우 매장 정보 리스트를 여러 개 저장하는 리스트 변수
    print(">>>>> 쿠우쿠우 매장 크롤링 중 >>>>>")
    qooqoo_store(result)
    # print(result)

if __name__ == "__main__" :
    main()





