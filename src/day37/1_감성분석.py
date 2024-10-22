# p.293 # python3.8

# 네이버 영화 리뷰 데이터
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 다운로드
train_file = tf.keras.utils.get_file(
    'ratings_train.txt',     # 다운로드된 파일의 이름 지정
    origin='https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt',       # 파일을 다운로드할 URL 지정
    extract=True        # 만약에 압축파일이면 자동으로 압축풀기 지정
)

# 2. 판다스 이용하여 해당 파일 객체로부터 파일 읽어오기 # \t 구분자
train = pd.read_csv(train_file, sep = '\t')

# 3. 읽어온 파일의 크기
print(train.shape)      # (150000, 3)
print(train.head())     # id : 게시물번호, # document : 리뷰명, # label : 0부정 1긍정
#          id                                           document  label
# 0   9976970                                아 더빙.. 진짜 짜증나네요 목소리      0
# 1   3819312                  흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나      1
# 2  10265843                                  너무재밓었다그래서보는것을추천한다      0
# 3   9045019                      교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정      0
# 4   6483659  사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 ...      1

# 4. 레이블 별 개수 # pd객체['필드명'].value_counts() : 지정한 필드의 데이터별 개수
cnt = train['label'].value_counts()
print(cnt)
# label
# 0    75173
# 1    74827
# Name: count, dtype: int64

# 5. 레이블 별 비율 시각화
sns.countplot(x='label', data=train)
plt.show()

# 6. 결측치(데이터 없는 / 빈 값) 확인 # pd객체.isnull()
print(train.isnull().sum())     # 결측치 개수 확인
# id          0
# document    5
# label       0
# dtype: int64
print(train[train['document'].isnull()])    # 결측치가 있는 'document'의 행 확인 # 추후에 결측치 행 삭제
#              id document  label
# 25857   2172111      NaN      1
# 55737   6369843      NaN      1
# 110014  1034280      NaN      0
# 126782  5942978      NaN      0
# 140721  1034283      NaN      0

# 7. 긍정/부정 텍스트 길이 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
data_len = train[train['label'] == 1]['document'].str.len()       # pd 객체 내 'label' 필드값이 1인 'document' 필드의 텍스트 길이 구하기
print(data_len)
ax1.hist(data_len)      # 히스토그램 차트
ax1.set_title('positive')

data_len = train[train['label'] == 0]['document'].str.len()
ax2.hist(data_len)
ax2.set_title('negative')
fig.suptitle('Number of characters')

plt.show()

# 8. 형태소 분석기 객체 불러오기 # 형태소란 : 의미를 가지는 가장 단위 단위 # 즉] 더이상 쪼갤 수 없는 최소의 의미 단위
# 오늘날씨어때 -> 오늘, 날씨 vs 오늘날 씨 - 이처럼 띄어쓰기가 안 돼 있을 때는 형태소 분석이 어렵다.
import konlpy   # konlpy 패키지 설치
from konlpy.tag import Kkma, Komoran, Okt   # Mecab 제외
kkma = Kkma()       # 객체 생성
komoran = Komoran() # 객체 생성
okt = Okt()     # 객체 생성
# Mecab 생략

# 9. 형태소별 샘플
text = "영실아안녕오늘날씨어때?"

def sample_ko_pos(text) :
    print(f"=== {text} ===")
    print("kkma:", kkma.pos(text))
    print("komoran:", komoran.pos(text))
    print("okt:", okt.pos(text))
    print("\n")

print(sample_ko_pos(text))

text2 = "영실아안뇽오늘날씨어때?"
print(sample_ko_pos(text2))

text3="정말 재미있고 매력적인 영화에요 추천합니다."
print(sample_ko_pos(text3))

# 10. 데이터 전처리
# 텍스트 전처리(영여와 한글만 남기고 삭제)
train['document'] = train['document'].str.replace("[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ]", "", regex=True)   # pd판다스버전 2.x이상부터, regax=True 속성. 정규표현식을 사용하겠다.
print(train['document'].head())

# 결측치/빈 데이터 제거 # .dropna() : 결측치 제거 함수
train = train.dropna()
print(train.shape)

# 스탑워드와 형태소 분석 (한글 불용어) # 불용어 제거
def word_tokenization(text) :
    # 불용어 목록 : 관사, 전치사, 조사, 접속사 등 의미가 없는 단어를 제거
    stop_words = ['는', '을', '를', '이', '가', '의', '던', '고', '하', '다', '은', '에', '들', '지', '게', '도']

    # 방법1
    '''
    list = []
    for word in okt.morphs( text ) :
        if word not in stop_words :
            list.append(word)
    return list
    '''

    # 방법2 : 컴프리헨션 # Open Korean Text (한국어 형태소 분석기 객체)
    # okt.morphs() : 분석결과를 리스트로 반환 vs okt.pos() : 분석결과를 튜플로 반환
    return [word for word in okt.morphs(text) if word not in stop_words]    # 리스트 컴프리헨션
    # 실습 : 문장이 15,000 개라서 시간이 걸린다
data = train['document'].apply((lambda x :word_tokenization(x)))       # document 열에 데이터 하나씩 불용이제거 함수에 대입한다
print(data.head())



# 11.
# data = train['document'].aplly((lambda x : word_tokenization(x)))
data = train['document']
print(data.head())

# train 과  validation 분할 # 훈련용과 테스트용 분할
training_size = 120000

# train 분할
train_sentence = data[:training_size]       # 0 ~ 110000
valid_sentence = data[training_size:]      # 110000 ~

# label 분할
train_labels = train['label'][:training_size]
valid_labels = train['label'][training_size:]

# 12. 단어 사전 만들기 # .fit_on_texts() : 문자와 숫자(인덱스)를 매칭한다 # 문자를 숫자로 변환한다
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# vocab_size 설정
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
print("총 단어 개수 : ", len(tokenizer.word_index))

# 5회 이상만 vocab_size에 포함
def get_vocab_size(threshold) :
    cnt = 0
    for x in tokenizer.word_counts.values():
        if x >= threshold:
            cnt = cnt + 1
    return cnt

vocab_size = get_vocab_size(5)  # 5회 이상 출현 단어
print("vocab_size: ", vocab_size)

# <OOV> : 사전에 없는 단어
oov_tok = '<OOV>'
vocab_size = 15000
tokenizer = Tokenizer(oov_token=oov_tok, num_words=vocab_size + 1)
tokenizer.fit_on_texts(data)

print(f'총 단어 개수 : {len(tokenizer.word_index)}')

# 14. 숫자 벡터로 변환
print(train_sentence[ : 2])
train_sequences = tokenizer.texts_to_sequence(train_sentence)
valid_sequences = tokenizer.texts_to_sequences(valid_sentence)
print(train_sequences[: 2])

# 15. 문장 중에서 최대 길이 구하기 # 모든 문자의 길이를 맞추기
# 모든 문장들이 길이가 일치하면 모델 성능 도움 # 최대 길이의 문자로 일치화
max_length = max(len(x) for x in train_sequences)
print(f"문장 최대 길이 : {max_length}")

# 16. 문장 길이를 동일하게 맞춘다 # 패딩
from tensorflow.keras.preprocessing.sequence import pad_sequences
trunc_type = 'post'     # 길이를 초과하는 자르는 속성 # post 뒤에 자르기
padding_type = 'post'   # 길이를 미달하는 경우 0으로 채우는 속성 # post 뒤에 채운다
train_padded = pad_sequences(train_sequences, truncating = 'post', padding=padding_type, maxlen=max_length)
valid_padded = pad_sequences(valid_sequences, truncating=trunc_type, padding = padding_type, maxlen=max_length)

import numpy as np
train_laebls = np.asarray(train_labels)     # 배열로 변환
valid_labels = np.assarray(train_labels)    # 배열로 변환
print(f'샘플 : {train_padded[ : 1]}')



# 17. 모델
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional

def create_model() :
    model = Sequential([
        Embedding(vocab_size, 32),
        Bidirectional(LSTM(32, return_sequences=False)),     # 양방향이면 유닛 뉴런 2
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # 이진 분류에서 자주 사용하는 활성화함수 # 츨력레이어
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

    model = create_model()
    print(model.summary())

    # 18. 학습
    history = model.fit(train_padded, tirain_labels,  # 훈련용
                        validataion_data(valid_padded, valid_labels),   # 학습중 사용할 테스트용
                        barch_size=64,      # 모델이 한 번에 처리할 데이터 수
                        epochs=10,          # 에포크
                        verbase=2           # 학습 시 콘솔에 요약 정도 0:출력없음 1:진행률바 2:결과요약만
                        )



# # 가장 좋은 loss의 가중치 저장
# checkpoint_path = 'bese_performed_model_ckpt'
# chekpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True,)

    print(history)  # 최종정확도, 손실함수 확인

    # 19. 새로운 리뷰 텍스트의 감정 분석 하기 # 예측하기
    new_reviews = ['영화 정말 재미있다', '정말 지루하다', '그냥 보통이었어요', '생각보다 재미가 없다']
    # 새로운 리뷰도 전처리
    new_sequences = tokenizer.texts_to_sequences(new_reviews)
    new_padded_sequences = pad_sequences(new_sequences, max_lengh=max_length)
    # 모델 이용한 감성 예측
    result = model.predict(new_padded_sequences)
    # 예측 결과
    for index, review in enumerate(new_reviews)    : # for 인덱스, 반복변수 in enumerate(반복할객체) :
        print(f'리뷰 : {review}, 확률 : {result[index]}')   # 0 ~ 1 사이의 비율 # 0.5초과 긍정 # 0.5미만 부정


