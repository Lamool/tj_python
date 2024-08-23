from flask import Flask     # (1)
app = Flask( __name__ )     # (2)

# ========== HTTP 매핑 ==========
# (((2)에서 보통 관례적으로 app을 쓴다)) app.route
# 대괄호 리스트
# @app.route(

# return : Flask에서 HTTP Response Content-Type : 파이썬의 리스트 타입, 딕셔너리 타입, 문자열 타입(JSON) 제공
@app.route("/", methods = ['GET'])   # vs spring @GetMapping()
def index1() :
    return "Hello HTTP method GET"

@app.route("/", methods = ['POST'])   # vs spring @PostMapping()
def index2() :
    return [3, 3]

@app.route("/", methods = ['PUT'])   # vs spring @PutMapping()
def index3() :
    return { 'result' : True }

@app.route("/", methods = ['DELETE'])   # vs spring @DeleteMapping()
def index4() :
    return "true"


# ==============================

if __name__ == "__main__" :   # (3)
    app.run(debug=True)       # debug=True    디버그[정보 또는 오류 콘솔 출력 제공] 모드

