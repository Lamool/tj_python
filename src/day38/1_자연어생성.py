import tensorflow as tf
import pandas as pd

# 1. 데이터 수집 # .get_file()
file = tf.keras.utils.get_file(
    'ratings_train.txt',    # 파일명
    origin = 'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt',    # 다운로드 받을 링크
    extract = True  # 압축 설정
)

df = pd.read_csv(file, sep = '\t')
print(df[1000:1007])    # 데이터 중 임의의 행 확인

# 2. 데이터 전처리
from konlpy.tag import Okt      # 한글 형태소 분석기 클래스 # Open Korean Text
okt = Okt()     # 파이썬객체생성 방법 : 클래스명() # vs Okt okt = new Okt()

def word_tokenization(text) :
    # 방법 1
    list = []
    result = okt.morphs(text)       # okt.morphs() : ((-> 리스트 반환))  vs okt.pos() : ((-> 튜플 반환))
    for word in result :
        list.append(word)
    return list
    # 방법 2
    # return [word for word in okt.morphs(text)]    # 리스트 컴프리헨션
def preprocessing(df) :
    df = df.dropna()    # 데이터프레임(df) # 결측값 제거
    df = df[1000:2000]  # 샘플 데이터 1000개 사용
    df['document'] = df['document'].str.replace("[^A-Za-z0-9가-힣-ㄱ-ㅎㅏ-ㅣ ]", "", regex=True)  # 데이터프레임(df)['열/속성 이름'] # 데이터프레임(df) 안에서의
    # "안녕하세요".replace()     : 문자열을 치환하는 함수
    # df['document'].replace() : 특정 열/속성의 여러 개 문자열 치환 함수 # str : 문자열 반환
    # * 서로 다른 객체들이 동일한 이름의 함수/기능 제공하는 경우 # 매개변수와 반환이 다를 수 있다
    data = df['document'].apply( (lambda x: [word_tokenization(x)] ))
    # data = df['document'].apply((lambda x: [word for word in okt.morphs(x)]))
    return data

'''
# 일반함수
def func1(param) :
    return param + 1
func1(2)
# 람다식함수
func2 = lambda param : param + 1
func2(2)
'''

review = preprocessing(df)
print(review)
print (review[:10])

# 트콘화 및 패딩
from tensorflow.keras.preprocessing.text import Tokenizer               # 토큰 관련 클래스
from tensorflow.keras.preprocessing.sequence import pad_sequences       # 패딩 관련 클래스
tokenizer = Tokenizer()     # 객체 생성

def get_tokens(review) :
    # 토큰객체.fit_on_texts() : 각 단어의 인덱스(숫자)를 대응하는 *단어사전* 생성 # 빈도수에 따라 인덱스(숫자)위치가 결정
    tokenizer.fit_on_texts(review)
    print(tokenizer.word_index)
    total_words = len(tokenizer.word_index) + 1
    print(total_words)
    # 각 문장을 숫자(벡터)로 변환 # .texts_to_sequences()
    tokenized_sentences = tokenizer.texts_to_sequences(review)   #
    print(tokenized_sentences)

    input_sequences = []
    for token in tokenized_sentences :
        for t in range(1, len(token)) :
            n_gram_sequence = token[:t+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words     # ((얘네 소괄호 생략되는 것. return 결과 한 개인데 얘네 ,로 묶여 있는 이거 튜플이다))

input_sequences, total_words = get_tokens((review))
input_sequences[31:40]  # n_gram으로 리스트된 데이터 샘플 확인

# 단어 사전
print(f"감동 : {tokenizer.word_index['감동']}")
print(f"영화 : {tokenizer.word_index['영화']}")
print(f"코믹 : {tokenizer.word_index['코믹']}")


# 4. 패딩 : 모델이 시퀀스(문장)들을 학습할 때 길이를 맞춤으로써 동일한 차원을 처리할 수 있게 하기 위해서 해야 한다. 문장의 길이 동일하게 맞추기
max_len = max([len(word) for word in input_sequences])
print("max_len : ", max_len)        # 가장 긴 문장은 59개 단어를 가졌다
# 패딩 함수를 이용한 패딩화 하기 # pad_sequences(데이터리스트, maxlen=최대길이, padding='pre앞post뒤')
result = pad_sequences(input_sequences, maxlen=max_len, padding='pre')
print(result)
import numpy as np  # 넘파이 객체
input_sequences = np.array(result)    # 패딩 결과를 다시 배열로 변환
print(input_sequences)

# 5. 독립변수와 종속변수(정답) 구분하기 : 모델의 학습을 위해서
from tensorflow.keras.utils import to_categorical
# x는 독립변수 데이터로, 각 시퀀스의 마지막 단어를 제외함(왜? 마지막 단어는 예측하기 위해)
# 즉, 모델은 시퀀스의 처음부터 마지막 단어 직전까지를 학습시킨다
# ((이게 무슨 말이야. 학습한 것을 바탕으로 마지막 단어(그러니까 맞는 단어. 정답? 같은 것을) 일부러 남겨두고 확인하겠다??))
x = input_sequences[ :, : -1]    # 마지막 값은 제외함
# y는 종속변수 데이터로, 각 시퀀스의 마지막 단어를 원핫인코딩으로 변환한다. (왜? 위치 찾기 위해서)
y = to_categorical(input_sequences[:, -1], num_classes=total_words)
a =to_categorical([0, 1, 2, 3], num_classes=4)

# 6. 모델 생성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout

embedding_dim = 256
model = Sequential()    # 딥러닝 모델
    # 1. 임베딩 레이어 (원핫인코딩(벡터) VS 밀집인코딩(벡터)) : 밀집 벡터로 변환 하는 역할
        # input_dim : 입력받을 단어의 총 개수
        # output_dim : 밀집 벡터로 변환된 벡터 차원수 #
model.add(Embedding(input_dim=total_words, output_dim=embedding_dim, input_length=max_len - 1))
    # 2. 양방향(Bidirectional), RNN알고리즘(LSTM)  # Bidirectional(LSTM()) : 양방향 RNN알고리즘
model.add(Bidirectional(LSTM(units=256)))   # 유닛/노드/뉴런 : 학습하면서 특징/파라미터/값을 저장하는 개수 # 양방향 * 2
    # 3. 출력레이어 : 출력개수는 2진분류가 아닌 다중분류이므로 활성화 함수는 'softmax',
model.add(Dense(units=total_words, activation='softmax'))   # 마지막 출력 레이어의 유닛/노드/뉴런의 종속변수 개수
    # 5. 컴파일 (머신러닝과 다르게 딥러닝은 학습도중에 손실함수(loss)와 평가지도(accuracy정확도) 확인/모니터링 할 수 있는 함수/기능
    # 최적의 파라미터 찾기 = 튜닝 작업
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x, y, epochs=20)    # X : 독립변수, Y : 종속변수 # 학습 수
# 딥러닝 : 머신러닝 보다 조금 더 복잡(다차원)하고 복잡한 학습을 함으로 패턴 찾기

# 7. 문장생성함수 / 예측
def text_generation(sos, count) :
    for _ in range (1, count) :     # 1부터 생성할 단어 수까지
        # sos(새로운 문장)을 벡터로 변환
        token_list = tokenizer.texts_to_sequences([sos])[0]
        # sos(새로운 문장)을 패딩화 해서 학습된 문장들과 동리일하게 일치
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        # 예측하기 # 모델객체명.pad_sequences(
        result = model.predict(token_list)
        print (result)            #

        predicted = np.argmax(result)  # numpy(np)넘파이객체 #  np.argmax( 배열  , axis = 1) 배열내 최대값 인덱스 찾기
        # 반복문을 이용한 단어 사전에서 비율이 높은 예측단어 찾기
        for word, index in tokenizer.word_index.items():  # 토큰나이저객체.word_index.items() : 단어 사전들의 단어
            if index == predicted:
                # 만약에 단어 사전내 인덱스가 예측 인덱스와 같으면
                output = word  # 찾은 인덱스의 단어 저장
                break
        sos += " " + output  # 새로운 문장 뒤에 예측한 단어 연결하기
    return sos

print(text_generation('연애 하면서', 12))  # '연애 하면서' 뒤로 12개의 예측 단어 붙여준다
print(text_generation('꿀잼', 12))
print(text_generation('최고의 영화', 12))
print(text_generation('손발 이', 12))


