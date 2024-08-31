import pandas as pd
import json

df = pd.read_csv('행정동 단위 서울 생활인구(내국인).csv', encoding='cp949', engine='python')
# print(df)
def totaldata():
    totaldata_tojson = df.transpose().to_json(force_ascii=False)
    # print(totaldata_tojson)
    result = json.loads(totaldata_tojson)
    # print(result)
    return result

def totalDescribe():
    total_describe =  df.describe().transpose().to_json(force_ascii=False)
    # print(total_describe)
    result = json.loads(total_describe)
    return result

# 시간대별 총 인구수 기술통계
def timeDescribe():
    time_describe = df.groupby('시간대구분')['총생활인구수'].describe().to_json()
    # print(time_describe)
    result = json.loads(time_describe)
    return result

# 시간대별 가장 인구수 많은 행정동코드
def highest_population():
    result = []
    # 데이터프레임 '시간대구분' , '총생활인구수' 내림차순으로 정렬 , ascending=False .sort_values() 에서 내림차순 하겠다는 뜻
    dataDf = df.sort_values(by=['시간대구분' , '총생활인구수'], ascending=False)
    # print(dataDf)
    # 정렬한 데이터프레임에서 특정 열만 뽑아 변수에 저장하기
    newDat = dataDf[['시간대구분' , '총생활인구수' , '행정동코드']]
    # print(result)
    for i in range(0 , 24):
        # 시간대 별로 인구수가 가장 많은 행 1개씩만 뽑아서 json 변환 후 리스트에 저장
        iData = newDat[newDat['시간대구분']== i][['시간대구분' , '총생활인구수' , '행정동코드']][0:1].to_json(orient='index')
        strData = json.loads(iData)
        print(strData)
        result.append(strData)
    # print(result)
    return result

# 행정동코드 별 남 여 인구수
def code_male():
    # 행정동 코드 별로 그룹한 후 모든 열 더하기
    data1 = df.groupby('행정동코드').sum()
    # print(data1)
    # 행정동 코드별로 그룹한 데이터에서 남자만 구하기 , like= 특정 문자가 포함된 axis=1 열 찾기
    data2 = data1.filter(like="남" , axis=1)
    # print(data2)
    malesum = data2.sum(axis=1)
    # print(malesum)
    malesum_to = malesum.to_json(orient='index')
    malesum_to_str = json.loads(malesum_to)
    # print(malesum_to_str)
    return malesum_to_str

# code_male()

def code_female():
    # 행정동 코드 별로 그룹한 후 모든 열 더하기
    data1 = df.groupby('행정동코드').sum()
    # print(data1)
    # 행정동 코드별로 그룹한 데이터에서 남자만 구하기 , like= 특정 문자가 포함된 axis=1 열 찾기
    data2 = data1.filter(like="여" , axis=1)
    # print(data2)
    femalesum = data2.sum(axis=1)
    # print(femalesum)
    femalesum_to = femalesum.to_json(orient='index')
    femalesum_to_str = json.loads(femalesum_to)
    # print(femalesum_to_str)
    return femalesum_to_str

def data_add():
    # key 값이 겹칠때 value 값 리스트로 만들어서 합치기
    data1 = code_male()
    data2 = code_female()
    # print(data1.items())
    result = {}
    for key, value in data1.items():    # 딕셔너리의 키랑 값 순회
        print(key , value)
        if key in result:               # 만약에 키가 result 안에 있는데
            if result[key] != value:    # result딕셔너리의 키가 값이 같지않으면
                result[key] = [result[key] , value] # result 키 안에 값 추가
        else:
            result[key] = value         # 아니면 key와 값 추가
    for key, value in data2.items():
        if key in result:
            if result[key] != value:
                result[key] = [result[key] , value]
        else:
            result[key] = value
    # print(result)
    return result
