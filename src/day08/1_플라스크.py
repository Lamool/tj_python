'''
    Flask
        - 파이썬으로 만들어진 웹 프레임워크 (vs JAVA spring)
        - Flask vs 장고
    - 설치
        (1) flask 모듈 설치
            방법1 : from flask에 커서를 두고 빨간 느낌표 클릭후 -> install
            방법2 : 상단메뉴 -> 파일 -> 설정 -> 왼쪽메뉴[프로젝트] 하위 [인터프리터]
                        -> [+] alt + insert -> Flask 검색 후 패키지 선택 -> [패키지 설치]
                - pip : 파이썬에서 패키지 소프트웨어를 설치/관리하는 시스템 (vs 그레이들)
        (2) Flask 모듈 가져오기
            from flask import Flask
        (3) Flask 객체 생성
            app = Flask(__name__)
        (4) Flask 프레임워크 실행
            if __name__ == '__main__' :
                app.run(debug=True)
    - # HTTP GET 매핑 설정
        @app.route('/')
'''
# Flask 모듈 가져오기
from flask import Flask
# Flask 객체 생성
app = Flask( __name__ )
# HTTP GET 매핑 설정
@app.route('/')
def index() : # 매핑 함수
    return 'Hello Flask'
# Flask 프레임워크 실행
if __name__ == '__main__' :
    app.run(debug=True)     # debug


# 콘솔 확인
# Flask port 확인 : http://127.0.0.1:5000
# 테스트 : (1) 크롬 웹 주소 http://127.0.0.1:5000  (2) Talend Api  (3) JS-AJAX
'''
 * Serving Flask app '1_플라스크'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 123-531-307
'''
