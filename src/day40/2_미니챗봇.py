# 2_미니챗봇.py
import numpy as np
import pandas as pd

# 현재 시간을 구하는 서비스 함수
def time_info() :
    print('현재 시간 서비스 실행')
    result = '3시'       # 현재 시간을 구하는 로직 하는 함수 호출 # 자바 REST 호출 함수0
    return f'현재 시간은 {result} 입니다.'      # 챗봇이 응답하는 메시지 구성 반환

def stock_info(*kwargs) :  # 여러 개의 매개변수를 받기
    # 가변길이의 매개변수 : *매개변수명(튜풀), **매개변수명(딕셔너린) # dto
    # ((* / ** 차이. 가변 길이))
    print("현재 재고 서비스 실행")
    result = 30     # 현재 땡땡의 제품 재고를 구하는 로직 하는 함수 호출 #
    if result == False :
        return "제품을 확인 하기 위해서 제품을 정확히 알려 주세요"
    return f'{"콜라"}의 재고는 {result} 입니다.'

    # ((제품명은 오타 나는 거 예측하면 안 된다. 클라로 입력 시 콜라를. 아니면 다른 방법? 동일한 제품 있는지))


# 예측한 확률의 질문과 함수 매칭 딕셔너리
response_functions = {
    2 : time_info,   # () 제외 # 지금 몇 시에요?라는 예측 질문을 찾았을 때 함수 실행
                    # ((함수 정의한 것을 대입한 것이기 때문에))
    5 : stock_info
}

# 1. 데이터 수집 # csv , db , 함수(코드/메모리)
data = [
    {"user": "안녕하세요", "bot": "안녕하세요! 무엇을 도와드릴까요?"},
    {"user": "오늘 날씨 어때요?", "bot": "오늘은 맑고 화창한 날씨입니다."},
    {"user": "지금 몇 시에요?", "bot": "현재 시간을 알려 드릴게요."},
    {"user": "좋은 책 추천해 주세요", "bot": "최근에 인기가 많은 책은 '파이썬 데이터 분석'입니다."},
    {"user": "고마워요", "bot": "천만에요! 더 필요한 것이 있으면 말씀해주세요."},
    {"user": "콜라의 재고를 알려 주세요", "bot" : "제품의 재고를 알려 드릴게요"}
]
data = pd.DataFrame( data ) #데이터프레임 변환
print( data )
# 2. 데이터 전처리
inputs = list( data['user'] ) # 질문
outputs = list( data['bot'] ) # 응답

from konlpy.tag import Okt
import re # 정규표현식
okt = Okt()
def preprocess( text ) :
    # 한글 과 띄어쓰기(\s) 를 제외한 문자 제거 # ^부정
    result = re.sub( r'[^가-힣\s]' , '' , text ) # 정규표현식 # 일반적인 문자열 정규표현식
    print( result )
    # 2. 형태소 분석
    result = okt.pos( result ); print( result ) # [ ('라면','Noun') , ('하다',Verb) ]
    # 3. 명사(Noun)와 동사(Verb) 와 형용사(Adjective) 외 제거
    # 형태소 분석기가 각 형태소들을 명칭하는 단어들 (pos)변수 존대한다.
    result = [ word for word , pos in result if pos in ['Noun','Verb','Adjective'] ]
    # 4. 불용어 생략
    # 5. 반환
    return " ".join( result ).strip( ) # strip() 앞뒤 공백 제거 함수
# 전처리 실행  # 모든 질문을 전처리 해서 새로운 리스트
processed_inputs = [ preprocess(질문) for 질문 in inputs ]
print( processed_inputs )
# ['안녕하세요', '오늘 날씨 어때요', '지금 몇 시', '좋은 책 추천 해 주세요', '고마워요']

# 3. 토크나이저
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts( processed_inputs ) # 전처리된 단어 목록을 단어사전 생성
print( tokenizer.word_index ) # 사전확인
# {'안녕하세요': 1, '오늘': 2, '날씨': 3, '어때요': 4, '지금': 5, '몇': 6, '시': 7, '좋은': 8, '책': 9 } ~~

input_sequences = tokenizer.texts_to_sequences( processed_inputs ) # 벡터화
print( input_sequences )
# ['안녕하세요', '오늘 날씨 어때요', '지금 몇 시', '좋은 책 추천 해 주세요', '고마워요']
# [[1], [2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12], [13]]

max_sequence_length = max( len(문장) for 문장 in input_sequences ) # 여러 문장중에 가장 긴 단어의 개수
print( max_sequence_length ) # '좋은 책 추천 해 주세요' # 5

input_sequences = pad_sequences( input_sequences  , maxlen=max_sequence_length ) # 패딩화 # 가장 길이가 긴 문장 기준으로 0으로 채우기
print( input_sequences ) #  '오늘 날씨 어때요' --> [ 2  3  4 ] --> [ 0 0 2 3 4 ] # 좋은 성능을 만들기 위해 차원을 통일

# 종속변수 # 데이터프레임 --> 일반 배열 변환
# output_sequences = np.array(outputs)
output_sequences = np.array(range(len(outputs)))
print( output_sequences )

# ((단어 사전을 만들고 / 만들어진 단어 사진 기반으로 벡터 / 모든 문자열의 길이를 일치화 하기 위해 모든 문장 단어의 개수를 구한다??)
# ((학습률을 올리기 위해 차원 통일?))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

# 1. 모델
model = Sequential()
    # input_dim : 입력받을 단어의 총 개수
    # output_dim : 밀집 벡터로 변환된 벡터 차원수
model.add(Embedding(input_dim=len(tokenizer.word_index), output_dim=50, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(64))),     # 256, 128, 64, 32
model.add(Dense(len(outputs), activation='softmax'))    # 종속변수의 값 개수는 응답 개수

# 2. 컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. 학습
model.fit(input_sequences, output_sequences, epochs=10)

# 4. 예측하기
def response(text) :
    # 1. 예측할 값도 전처리 한다
    text = preprocess(text)
    print(text)

    # 2. 예측할 값도 토큰과 패딩 # 학습된 모델과 데이터 통일
    text = tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, maxlen=max_sequence_length)

    # 3. 예측
    result = model.predict(text)

    # 4. 결과 # 가장 높은 확률의 인덱스 찾기
    max_index = np.argmax(result)

    msg = outputs[max_index]        # 일반적인 메시지

    # 만약에 예측한 질문의 인덱스가 함수 매칭 딕셔너리내 존재하면
    if max_index in response_functions :
        msg += response_functions[max_index]()      # 함수호출

    # 5.
    return outputs[max_index]

# 확인
print(response('안녕하세요'))    # 질문이 '안녕하세요', 학습된 질문 목록 중에 가장 높은 예측비율이 높은 질문의 응담을 출력한다
# 서비스 제공한다 # 플라스크
while True :
    text = input('사용자 : ')
    result = response(text)
    print(f'챗봇: {result}')


