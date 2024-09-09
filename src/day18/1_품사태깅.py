'''
    형태소 : 의미가 있는 가장 작은 단위
    단어 : 의미를 갖는 문장의 가장 작은 단일 요소, 문장 내에서 분리 될 수 있는 부분 # 형태소, 접사 구분
    품사 태깅 : 형태소의 뜻과 문맥을 고려하여 품사를 붙인 것
    품사 태깅 패키지 : konlpy (자바 기반의 소프트웨어, 자바 설치가 필요함)
'''
# 1. 모듈 호출
from konlpy.tag import Okt     # konlpy 설치
    # 1. jvm 버전 확인 : cmd --> java -version 입력 후 확인

# 2. 분석할 한글
text = "나는 사과를 먹었다"

# 3. 품사 태깅
    # 3-1. 품사 태깅 객체 생성
okt = Okt()
    # 3-2. 품사 태깅 함수 실행 # okt.pos(분석할문장)
tag_words = okt.pos(text)

# 4. 확인
print(tag_words)    # [('나', 'Noun'), ('는', 'Josa'), ('사과', 'Noun'), ('를', 'Josa'), ('먹었다', 'Verb')]
