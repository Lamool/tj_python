from flask import Flask

app = Flask( __name__ )

# [모듈] controller.py의 매핑 함수들 가져오기 (( -> if __name__ == "__main__" : 보다는 위에?))
from controller import *    # ((매핑 때문에 웬만하면 *로 쓸 것))

if __name__ == "__main__" :
    app.run()


