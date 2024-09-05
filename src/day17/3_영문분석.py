import pandas as pd
import glob
import re
from functools import reduce
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud

# 최초 1번만 실행
# import nltk
# nltk.download()

# [1] 여러 개의 파일명을 불러오기 # glob.glob() 특정 패턴과 일치한 파일명을 모두 찾는 함수
all_files = glob.glob('exportExcelData_2024090514*.xls')
# print(all_files)

# [2] 여러 개 파일명에 해당하는 엑셀파일을 호출해서 pd로 가져오기
all_files_data = []
for file in all_files :     # 모든 파일명을 하나씩 반복한다
    # print(file)
    data_frame = pd.read_excel(file)
    # print(data_frame)
    all_files_data.append(data_frame)       # 불러온 엑셀 df를 리스트에 담는다

# print(all_files_data)       # 여러 개 df가 존재한다

# [3] 데이터프레임 합치기   # .concat (여러 개 프레임이 저장된 리스트, axis = 0(세로)1(가로))
all_files_data_concat = pd.concat(all_files_data, axis=0, ignore_index=True)
    # ignore_index=True : 인덱스를 0부터 새로 생기게 하겠다 - 0 ~ 499로. (합치기 전의 인덱스를 유지할 게 아니라. 0 ~ 99, 0 ~ 99)
# print(all_files_data_concat)

# [4] 데이터프레임을 csv로 변환/내보내기
all_files_data_concat.to_csv('riss_ai.csv', encoding='utf-8', index=False)
    # index=False : index를 포함시키지 않고 데이터를 저장하겠다

# [5] 데이터프레임의 제목(열)만 추출
all_title = all_files_data_concat['제목']
# print(all_title)

# [6] 단어 토큰화 준비
    # stopwords.words('english') : '영어' 불용어 목록 가져오는 함수
    # WordNetLemmatizer() : 표제어 추출기 객체 생성
        # 표제어 : 단어의 원형(기본형) 찾는 과정, running -> run, better -> good 변환   # 시제, 단/복수/ 진행어 등등 일반화 과정
영문불용어목록 = stopwords.words('english')
# print(영문불용어목록)
표제어객체 = WordNetLemmatizer
# print(표제어객체)

# [7] 단어 토큰화
words = []
for title in all_title :        #제목 목록에서 제목 하나씩 반복하기
    print(title)
    # 7-1. 영문이 아닌 것을 정규표현식을 이용해서 치환
    EnWords = re.sub(r'[^a-z]')
    # re.sub(pattern, replace, text) : text 중 pattern에 해당하는 부분을 replace로 대체한다.

