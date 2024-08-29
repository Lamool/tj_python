# p.124

# [1] 모듈 가져오기
import pandas as pd
import matplotlib.pyplot as plt

# [2] 리스트를 이용하여 DataFrame 생성
df = pd.DataFrame([[500, 450, 520, 610],
                         [690, 700, 820, 900],
                         [1100, 1030, 1200, 1380],
                         [1500, 1650, 1700, 1850],
                         [1990, 2020, 2300, 2420],
                         [1020, 1600, 2200, 2550]],
                    index = [2015, 2016, 2017, 2018, 2019, 2020],
                    columns = ['1분기', '2분기', '3분기', '4분기'])
# print(df)

# [3] DataFrame을 CSV 파일로 저장
df.to_csv('data.csv', header='False')        # df 객체를 csv로 내보내기

# [4] CSV를 DataFrame으로 불러오기
df2 = pd.read_csv('data', encoding='utf-8', index_col=0)
# print(df2)

# [5] 데이터프레임 객체를 JSON으로 가져오기
jsonResult = df2.to_json(orient='records', force_ascii=False)



# 1. 차트에 표시할 데이터


# print(df2.head(1))
# print(df2.columns)
# list = []
# list = df2.head(1)
# print(jsonResult)
# print(df2.index)
# aa= df2.index.data
# print(aa)



