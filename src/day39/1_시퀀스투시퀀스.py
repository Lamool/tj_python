# python3.8 gpu

import pandas as pd

# 1. 데이터수집
# 질문과 답변이 있는 말뭉치(대화내용)를 가져오기
    # pd.read_csv('csv로컬경로/웹경로')
    # Q(질문),A(답변),label(0:일상다반사,1:부정 2:긍정)
corpus = pd.read_csv('https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv')
# 확인
print(corpus['Q'].head())   # 질문 '열'의 상단 5개 데이터 확인
print(corpus['A'].head())   # 답변 '열'의 상단 5개 데이터 확인

# 확인 : 같은 인덱스로 Q와 A로 구성됨
print(f"Q {corpus['Q'][0]}")    # Q 12시 땡!
print(f"A {corpus['A'][0]}")    # A 하루가 또 가네요.

# 확인
print(corpus.shape)     # (11823, 3)    # (행, 열)    # .shape : 데이터프레임 객체의 차원 확인 속성

# 샘플링(1,000개 사용)
texts = []  # 질문 리스트
pairs = []  # 답변 리스트

# for index, value in enumerate(리스트/튜플) :
# for value in 리스트/튜플 :
print(zip(corpus['Q'], corpus['A']))   # zip(리스트, 리스트)
print(type(zip(corpus['Q'], corpus['A'])))

for Q, A in zip(corpus['Q'], corpus['A']):
    print(f'Q : {Q}, A : {A}')
for i, (text, pair) in enumerate(zip(corpus['Q'], corpus['A'])):
    texts.append(text)
    texts.append(pair)
    if i >= 1000 :  # 1000개의 인덱스만 사용 # 샘플링 # RAM문제
        break

# 2. 데이터 전처리 함수(정규식(regex expression)활용)
import re   # 파이썬 문자열 정규표현식 객체 ((pandas에서 사용하는 정규표현식도 있는데 둘이 다르다))

# 정규표현식
def clean_sentence(sentence):
    # 한글, 숫자를 제외한 모든 문자는 제거합니다.
    sentence = re.sub(r'[^0-9ㄱ-ㅎㅏ-ㅣ가-힣\s]', r'', sentence)    # 파이썬 내장용 문자열 정규표현식 함수
    # 1. re.sub(r'정규표현식', r'대체할문자', 문자열) : 파이썬 내장용 문자열 정규표현식 함수
    # 2. pd.str.replace("정규표현식", "", regex=True)   # 데이터프레임내 정규표현식 방법 # ((regex=True 이거 2? 버전 이상에서는 써줘야.))
    return sentence

# 전처리 함수 테스트
print(clean_sentence('안녕하세요~:'))
print(clean_sentence('텐서플로^@^%#@!'))

# konlpy 설치
# import konlpy

# 한글 형태소 분석기 # Okt 형태소
from konlpy.tag import Okt
okt = Okt()     # 한글 분석기 객체
def process_morph(sentence):
    return ' '.join(okt.morphs(sentence))       # 형태소 분석 결과 목록을 하나의 문자열 합치기

print('안녕하세요'.join(['유재석', '강호동']))     # 유재석안녕하세요강호동

# 질문전체, 답변시작, 답변끝
# 한글 문장 전처리
def clean_and_morph(sentence, is_question=True):    # 매개변수명=초기값 : 매개변수에 초기값 넣기
    # 한글 문장 (정규표현식 함수 실행)
    sentence = clean_sentence(sentence)
    # 형태소 변환 (형태소 변환 (형태소 함수 실행)
    sentence = process_morph(sentence)
    # 질문이 Question인 경우, Answer인 경우를 구분하여 처리
    if is_question:
        return sentence
    else :
        # 프로그래밍 언어에서 함수는 무조건 리턴(결과) 1개 이다/
        # START 토큰은 decoder input에 END 토큰은 decoder output에 추가
        return(f'<START> + {sentence}', f'{sentence}' + '<END>')


def preprocess(texts, pairs):
    questions = []      # 인코더에 입력할 질문 전체 레스트
    answer_in = []
    answer_out = []

    # 질의에 대한 전처리
    for text in texts:
        # 전처리와 morph 수행
        question = clean_and_morph(text, is_question=True)
        questions.append(question)

    # 답변에 대한 전처리
    for pair in pairs:
        # 전처리와 morph 수행
        in_, out_ = clean_and_morph(pair, is_question=False)
        answer_out.append(out_)

    return questions, answer_in, answer_out

(questions, answer_in, answer_out) = preprocess(texts, pairs)
print(questions[:2])
print(answer_in[:2])
print(answer_out[:2])

# 전체 문장을 하나의 리스트로 만들기
all_sentences = questions + answer_in + answer_out
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
warnings.filterwarnings('ignore')   # 경고 무시

# 단어 사전 만들기
    # filters='' : 토큰화 할 때 특정 기호를 제거(필터)    # 필터링 하지 않겠다는 뜻
    # lower = False : 토큰화 할 때 소문자로 변환할지 여부  # 기본값 true이므로 모든 영문을 소문자로 변환
        # false이므로 변환하지 않는다
    # oov_token : 단어 사전에 없는 단어를 매칭할 때 그 단어를 대체할 문자 <OOV> 표현
tokenizer = Tokenizer(filters='', lower=False, oov_token='<OOV>')   # 변수명 = 클래스명()
tokenizer.fit_on_texts(all_sentences)
print(tokenizer.word_index)     # 단어 사전 확인
print(len(tokenizer.word_index))        # 단어의 총 개수

# 등록된 사전에 따라 문장의 단어들을 벡터(숫자) 매칭하여 변환
questions_sequence = tokenizer.texts_to_sequences(questions);       print(questions_sequence[0])
questions_sequence = tokenizer.texts_to_sequences(questions);       print(questions_sequence[0])
questions_sequence = tokenizer.texts_to_sequences(questions);       print(questions_sequence[0])

# 패딩 (문장의 길이 맞추기 : 학습 데이터들의 차원 일치화 함으로 모델 성능 향상 )
MAX_LENGTH = 30 # 문장내 최대 단어 개수(길이) 는 임의로 30 # post : 빈칸을 뒤에 0으로 채우기
question_padded = pad_sequences( question_sequence , maxlen=MAX_LENGTH , padding='post' )
answer_in_padded = pad_sequences( answer_in_sequence , maxlen=MAX_LENGTH , padding='post' )
answer_out_padded = pad_sequences( answer_out_sequence , maxlen=MAX_LENGTH , padding='post' )
print( question_padded.shape , answer_in_padded.shape , answer_out_padded.shape ) # (1001, 30) (1001, 30) (1001, 30)

# 상속 : 하나의 클래스가 다른 클래스에게 속성/필드 과 함수/기능 물려두는 행위
    # 자바 : class 클래스A extends 클래스B{ }
        # this , super
    # 파이썬 : class 클래스A( 클래스B ) :
        # self , super
from tensorflow.keras.layers import Embedding , LSTM , Dense , Dropout
from tensorflow.keras.models import Model
# - 텐서플로의 Model 클래스로부터 상속받아 인코더 클래스 정의하기
class Encoder( tf.keras.Model ) :
    # 초기화함수 # 생성자 # 사용할 변수 , 레이어를 미리 불러와서 파라미터 값들을 미리 설정 한다.
    def __init__(self , units , vocab_size , embedding_dim , time_steps ):
                # units 매개변수1 : LSTM에서 사용할 유닛/노드/뉴런 수
                    # "안녕하세요, 오늘 날씨 어떄요?" 문장 가정이라고 했을때.
                # vocab_size 매개변수2 : 임베딩 레이어의 입력으로 들어가는 단어 크기
                    # "안녕하세요" , "오늘" , "날씨" ,"어때요" => 4
                # embedding_dim 매개변수3 : 임베딩 레이어의 각 단어를 크기의 벡터 차원
                    # 밀집행렬를 처리할때 한 단어를 표현을 차원수 # "안녕하세요" 몇차원으로 구성할지
                # time_steps 매개변수4 : 임베딩 레이어의 입력으로 들어가는 시퀀스의 길이
                    # 한번에 몇개의 단어를 모델이 학습하고 기억할지 단위 길이 # 2 => "안녕하세요" , "오늘"
        super( Encoder , self ).__init__() # 상속받은 슈퍼클래스의 초기화함수(생성자) 를 호출
        # 1. 임베딩 레이어
        self.embedding = Embedding( vocab_size , embedding_dim , input_length=time_steps )
        # 2. 드롭아웃 레이어 # 일반드롭아웃 # 0.2 : 20%를 무작위로 비활성
        self.dropout = Dropout( 0.2 )
        # 3. LSTM 레이어 #
        self.lstm = LSTM( units , return_state= True )

    # 실행 함수 # call
    def call(self, inputs):
        x = self.embedding(inputs)      # 임베딩 레이어에 따른 밀집행렬(벡터) 하기
        x = self.dropout(x)             # 드롭아웃 레이어에 따른 무작위 노드를 제외 하기
        x, hidden_state, call_state = self.lstm(x)      # LSTM 레이어에 따른 학습
            # x : LSTM 알고리즘이 특정 단어로부터의 특징(정보/패턴) 값
            # 은닉 상태 : LSTM 알고리즘이 현재 시점에서의 기록한 특징들(정보/패턴)을  저장하는 메모리
            # 셀 상태 : LSTM 알고리즘이 전체 단어들에서 중요한 특징(정보/패턴)들을 저장하는 메모리

            # (특징/패턴) 분석
                # CNN : 이미지 분석  # 곡선, 색감, 사이즈 ,비율, 질감(텍스처) 등등 # 0~255
                # RNN : 텍스트 분석, # 빈도, 감정, 형태소(동사, 형용사 등등, 단어의 의미 # 벡터

        # Dense 레이어가 없는 이유는 현재 클래스(인코더)의 목적은 입력과정 하기 위해서 --> 디코더 전달할 예정

        # ((임베디드에서 이야기하는 벡터랑 LSTM 벡터랑 다른?))
        #
        return [hidden_state, call_state]

# 텐서플로의 Model 클래스로부터 상속받아 디코더 클래스 정의하기
# 디코더
class Decoder(tf.keras.Model):
    # 1.
    def __init__(self, units, vocab_size, embedding_dim, time_steps) :
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, input_length=time_steps)

        self.dropout = Dropout(0.2)
        self.lstm = LSTM(units, return_state=True, return_sequences=True)
            # return_state=True : 생략가능, 은닉상태와 셀 상태 반환 설정
            # return_sequences=True : 모든 시점의 출력을 반환한다

        self.dense = Dense(vocab_size, activation='softmax')    # 최종 출력 레이어

    # 2.
    def call (self, inputs, initial_state):
        x = self.embedding(inputs)
        x = self.dropout(x)
        x, hidden_state, cell_state = self.lstm(x, initial_state=initial_state) # LSTM 레이어에 따른 학습
            # initial_state : 초기화상태 속성 # 인코더와 결합 이후에 인코더에 생성한 은닉상태와 셀 상태를 대입한다
        x = self.dense(x)       # 출력 레이어 # 출력 : 학습된 모델에서의 최종 출력된 값 : X
        return (x, hidden_state, cell_state)    # (최종확률값, 은닉상태, 셀상태)



# 모델 결합
# class Seq2Seq(tf.keras.Model):




