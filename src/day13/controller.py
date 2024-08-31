from app import app
from service import *

@app.route("/total" , methods=['get']) # http://localhost:5000/total
def getTotal():
    result = totaldata()
    # print(result)
    return result

@app.route("/describe" , methods=['get']) # http://localhost:5000/total
def getDescribe():
    result = totalDescribe()
    # print(result)
    return result

@app.route("/timedescribe" , methods=['get']) # http://localhost:5000/total
def getTimeDescribe():
    result = timeDescribe()
    # print(result)
    return result

@app.route("/highestpopulation" , methods=['get']) # http://localhost:5000/total
def gethighest_population():
    result = highest_population()
    print(result)
    return result

@app.route("/codemalefemale" , methods=['get']) # http://localhost:5000/total
def code_male_female():
    result = data_add()
    return result