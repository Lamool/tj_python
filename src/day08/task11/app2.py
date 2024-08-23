'''
    삼성전자주가.csv 파일의 정보를 테이블 형식으로 localhost:8080/index3.html 출력하시오.
        1. csv 파일을 읽어서 한 줄씩 딕셔너리로 변환 후 리스트 담기 (객체 쓰지 말 것, 제목 담지 말기 불필요한 거 제외)
        2. 플라스크 이용한 HTTP 매핑 정의 하기
        3. 스프링 서버에서 AJAX를 이용한 플라스크 서버로부터 삼성전자주가 정보 응답받기
'''

from flask import Flask         # (1) Flask 모듈 가져오기

app2 = Flask(__name__)           # (2) Flask 객체 생성
from flask_cors import CORS     # (3) CORS 모듈 가져오기
CORS(app2)                       # (4) 해당 Flask 객체 내 모든 HTTP에 대해 CORS 허용

from controller2 import *

if __name__ == "__main__" :     # (5) Flask 실행
    app2.run(debug=True)
