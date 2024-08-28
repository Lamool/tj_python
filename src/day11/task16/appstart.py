# 1. 플라스크 모듈 가져오기
from flask import Flask
# 2. 플라스크 객체 생성
app = Flask(__name__)

# 5. CORS 허용, 서로 다른 port 간의 데이터 통신 허용, 보안상 문제 발생할 수 있다
from flask_cors import CORS
CORS(app)       # 모든 HTTP에 대해 CORS 허용

# 4. controller 모듈 가져오기
from controller import *

# 3. 플라스크 웹 실행
if __name__ == '__main__' :
    app.run(host='0.0.0.0', debug=True)



