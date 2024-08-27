# 1. 모듈
from bs4 import BeautifulSoup
import urllib.request

result = [] # 할리스 매장정보 리스트를 여러 개 저장하는 리스트 변수, 2차원 리스트
for page in range(1, 51) :  # 1부터 50까지 반복
    # 할리스 매장 정보 url
    url = f"https://www.hollys.co.kr/store/korea/korStore2.do?pageNo={page}"    # 할리스 매장 정보 url
    response = urllib.request.urlopen(url)
    htmlData = response.read()
    soup = BeautifulSoup(htmlData, "html.parser")       # print(soup)
    tbody = soup.select_one('tbody')        # print(tbody)

    for row in tbody.select('tr') :
        # print(row)
        tds = row.select('td')
        store_sido = tds[0].string;     # print(store_sido)
        store_name = tds[1].string;     # print(store_name)
        store_address = tds[3].string;  # print(store_address)
        store_phone = tds[5].string;    # print(store_phone)

        store = [store_name, store_sido, store_address, store_phone]
        result.append(store)    # 리스트 안에 리스트요소 추가 : 2차원 리스트

print(result)

