#task12 -> app.py

from flask import Flask
app=Flask(__name__)
from flask_cors import  CORS
CORS(app)

#매핑
from controller import *

if __name__=="__main__":
    app.run(debug=True)