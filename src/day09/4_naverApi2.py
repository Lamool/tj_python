# 연습문제 : 2_naverApi.py 참고하여 네이버api로 블로그에서 입력 받은 값에 대한 검색 결과를 크롤링하여 JSON 파일로 저장하시오.

import json
import urllib.request

# 네이버 개발자 센터에서 애플리케이션 신청 후 발급받은 키와 비밀번호
네이버키 = 'cWeB3UhaS0qhN_AqW3J8'
네이버비밀번호 = 'wrPIy2DPlR'

# 지정한 URL의 요청을 실행하고 응답을 받는 함수 [code 2]
def getRequestUrl(url) :
    요청객체 = urllib.request.Request(url)                  # 2. 지정한 URL 설정
    요청객체.add_header("X-Naver-Client-Id", 네이버키)            # 3. HTTP 요청 객체 내 HEADER 정보 추가
    요청객체.add_header('X-Naver-client-Secret', 네이버비밀번호)     # 4. http요청시 네이버 api id와 비밀번호 전송

    try :   # 예외처리
        응답객체 = urllib.request.urlopen(요청객체)     # 5. 지정한 url 실행후 응답 반환
        print(f'>>code2 요청 URL 결과 상태 : {응답객체.getcode()}')   # 확인
        if 응답객체.getcode() == 200 :                    # 6. 만약에 응답의 상태가 2xx이면 성공
            return 응답객체.read().decode('utf-8')        # 7. 실행된 URL 내 모든 내용물 읽어오기
    except Exception as e :
        return None             # 8. 없으면 None


# [code 3] 매개변수로 검색대상, 검색어, 시작번호, 한 번에 표기할 개수를 받아서
# URL 구성하여 getRequestUrl() 메소드에게 요청하여 응답객체를 받아서 JSON 형식으로 반환 함수
def getNaverSearch(node, srcText, page_start, display) :
    base = "https://openapi.naver.com/v1/search"    # 1. 요청 url의 기본 주고
    node = f'/{node}.json'  # 2. 요청 url의 검색 대상과 json 파일이름
        # https://openapi.naver.com/v1/search/news.json
    parameters = f'?query={urllib.parse.quote(srcText)}&start={page_start}&display={display}'   # 3. 요청 url의 파라미터

    url = base + node + parameters  # 4. url 합치기
    responseDecode = getRequestUrl(url)     # 5. url 요청을 하고 응답 객체 받기,   [code 2]
    print(f'>>code3 요청URL : {url}')     # 확인
    if responseDecode == None : return None     # 6. 만약에 url 응답 객체가 없으면 None 반환
    else : return json.loads(responseDecode)    # 7. 응답객체가 있으면 JSON 형식으로 변환
        # json.loads(문자열) : JSON 형식으로 변환


# [code 4]
def getPostData(post, jsonResult, cnt) :
    # 응답받은 객체의 요소들
    title = post['title']
    link = post['link']
    description = post['description']
    bloggername = post['bloggername']
    blogger_link = post['bloggerlink']
    postdate = post['postdate']

    # 딕셔너리 생성
    dic = {
           'cnt' : cnt,
           'title' : title,
           'link': link,
           'description' : description,
           'bloggername' : bloggername,
           'blogger_link': blogger_link,
           'postdate': postdate,
          }
    jsonResult.append(dic)


# [code 1]
def main() :
    node = 'blog'   # 1. 크롤링할 대상[네이버 제공하는 검색 대상 : 1. news 2. blog 등등] - 공문 참고
    srcText = input('검색어를 입력하세요 : ')        # 2. 사용자 입력으로 받은 검색어 변수
    cnt = 0             # 3. 검색 결과 개수
    jsonResult = []     # 4. 검색 결과를 정리하여 저장할 리스트 변수

    # 5. 1부터 100까지의 검색 결과를 처리한다     # [code 2] 네이버 블로그 검색 결과에 대한 응답을 저장하는 객체
    jsonResponse = getNaverSearch(node, srcText, 1, 100)    # 5. [code 2]
    print(f'>>jsonResponse : {jsonResponse}')

    total = jsonResponse['total']       # 6. 전체 검색 결과 개수
    print(total)

    # 7. 응답객체가 None이 아닌면서 응답객체의 display가 0이 아니면 무한반복, url 응답객체가 없을 때까지
    while ((jsonResponse != None) and (jsonResponse['display'] != 0)) :
        # 8. 검색 결과 리스트(items)에서 item(post) 호출
        for post in jsonResponse['items'] :
            cnt += 1    # 응답 개수 변수 1증가
            # 9. [code 3]
            getPostData(post, jsonResult, cnt)
        # 10. start를 display만큼 증가시킨다
        start = jsonResponse['start'] + jsonResponse['display']
        # 첫 번째 요청 1, 100, 두 번째 요청 101, 100, 세 번째 요청 201, 100
        # 무료버전 기준으로 : start : 1001 오류가 발생하면서 종료된다. 1001 이상 하기 위해서는 API 유료 계약 해야 한다.
        jsonResponse = getNaverSearch(node, srcText, start, 100)

    print(f'전체 검색 : {total}건')
    print(f'가져온 데이터 {cnt}건')
    # print(jsonResult)   # 확인

    # (방법1) PY 객체를 JSON형식(문자열) 변환 후 파일 처리   # with open() as 파일변수명 : # with 종료시 자동으로 파일 닫기
    with open(f'{srcText}-naver-{node}.json', 'w', encoding='utf-8') as file:
        jsonFile = json.dumps(jsonResult, indent=4, sort_keys=True, ensure_ascii=False)
        file.write(jsonFile)  # 파일 쓰기


if __name__ == "__main__" :
    main()      # [code 1] 메소드 실행