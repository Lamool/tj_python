# 1. seaborn 라이브러리에 내장된 '타이타닉' 데이터를 가져오기
import seaborn as sns
titanic = sns.load_dataset('titanic')
print(titanic)

# 2. 호출된 '타이타닉' 데이터를 csv 파일로 저장
titanic.to_csv('타이타닉.csv', index=True)

# 3. 결측값(누락된 값/공백)
print('--------------------------------')
print(titanic.isnull().sum())   # 결측값 확인

# 4. 결측값을 치환, # fillna() null (결측)값 특정 값으로 채워주는 함수
    # (1) age 열의 결측값을 중앙값 (크기순으로 정렬된 상태에서 중간에 위치한 값 뜻)으로 치환
    # median() : 중앙값 반환해주는 함수
titanic['age'] = titanic['age'].fillna(titanic['age'].median())
print('--------------------------------')
print(titanic.isnull().sum())   # 확인 age에 결측값이 없어졌다.
    # (2) embarked 열의 결측값을 최빈값(집합의 빈도가 가장 많은 값)으로 치환
print('--------------------------------')
print(titanic['embarked'].value_counts())
titanic['embarked'] = titanic['embarked'].fillna('S')
print('--------------------------------')
print(titanic.isnull().sum())     # 확인 embarked에 결측값이 없어졌다
    # (3) embark_town 열의 결측값을 최빈값으로 치환
print('--------------------------------')
print(titanic['embark_town'].value_counts())
titanic['embark_town'] = titanic['embark_town'].fillna('Southampton')
print('--------------------------------')
print(titanic.isnull().sum())     # 확인 embark_town에 결측값이 없어졌다
    # (4) embark_town 열의 결측값을 최빈값으로 치환
print('--------------------------------')
print(titanic['deck'].value_counts())
titanic['deck'] = titanic['deck'].fillna('C')
print('--------------------------------')
print(titanic.isnull().sum())     # 확인 deck에 결측값이 없어졌다

print('--------------------------------')
print(titanic.info())   # 데이터의 기본 정보
print('--------------------------------')
print(titanic.survived.value_counts())




# 차트를 그려 데이터를 시각적으로 탐색하기
# 1.
import matplotlib.pyplot as plt

# plt.subplots(1, 2, figsize = (10, 5))       # ((한 행에 열 두 개를. 근데 이렇게 하면 뭐가 왼쪽이고 뭐가 오른쪽인지 몰라서 아래처럼))
f,ax = plt.subplots(1, 2, figsize = (10, 5))    # 서브플롯 이용한 한 번에 여러 개 플롯 띄우기

# 성별이 남자인 생존자여부
print(titanic['survived'][titanic['sex'] == 'male'].value_counts())

# 원형차트((plot.pie?)) 구성 # autopct : 원형차트 내 각 조각 백분율 표시 # ax = ax[0] : 첫 번째 자리 # explode = [0, 0.1] : 두 번째 조각을 10% 정도 떨어뜨리기 # shadow : 차트 그림자
titanic['survived'][titanic['sex'] == 'male'].value_counts().plot.pie(explode = [0,0.1], autopct = '%1.1f%%', ax = ax[0], shadow = True)
titanic['survived'][titanic['sex'] == 'female'].value_counts().plot.pie(explode = [0,0.1], autopct = '%1.1f%%', ax = ax[1], shadow = True)
ax[0].set_title('Survived (Male)')
ax[1].set_title('Survived (Female)')
plt.show()      # 남자 승객 생존율은 18.9%, 여자 승객은 74.2% 확인


# (객실) 등급별 생존자 수를 차트로 나타내기 # x축 : 등급(속성명) hue=생존자여부(속성명) # 등급별 생존자 여부 개수 # data = 데이터프레임
sns.countplot(x = 'pclass', hue = 'survived', data = titanic)
plt.title('Pclass vs Survived')
plt.show()      # 생존자는 1등급에서 가장 많고 사망자는 3등급이 가장 많다
    # 예) 동행 여부 속성(alone) 따라 생존자 수
sns.countplot(x = 'alone', hue = 'survived', data = titanic)
plt.show()      # 혼자 왔을 때 사망자가 더 많다



# 상관 분석 # 연속형 데이터만 가능 # 회귀 분석과 다른점 : 예측치 없다
    # 연속형 데이터만 가능하므로 연속형 데이터 열만 추출 # .select_dtypes(include=[타입1, 타입2])
    # ((지금 이 데이터에는 문자도 있어서.))
titanic2 = titanic.select_dtypes(include=[int, float, bool])
# 연속형 데이터만 존재했을 때 상관분석 실행
titanic_corr = titanic2.corr(method='pearson')      # ((pearson을 넣어주면 2차원 테이블?))
print(titanic_corr)

# 주석 풀기
# # 상관계수 : 0 ~ 1 정도와 방향을 하나의 수치 요약 # 0 관계가 거의 없다 ~ 1 관계가 강하다
#     # 양의 상관관계는 한 변수가 증가하면 다른 변수도 증가한다.
#     # 음의 상관관계는 한 변수가 증가하면 다른 변수도 감소한다.
# # 분석 : 남자성인은 생존과 음의 상관관계, 객실등급은 생존여부와 음의 상관관계, 혼자 탑승한 경우 음의 상관관계
#     # 남자가 증가하면 생존여부 감소한다, 객실등급이 증가하면 생존 여부가 감소한다.
#
# # 상관계수를 csv 저장
# titanic_corr.to_csv('타이타닉상관계수표.csv', index=True)
#
#
# # 특정한 변수 사이의 상관 계수 추출
# # ((survived와 adult_male가 만나는 지점의 값))
# print(titanic['survived'].corr(titanic['adult_male']))      # -0.5570800422053259
#               # 종속변수                  # 독립변수
# print(titanic['survived'].corr(titanic['fare']))            # 0.2573065223849622
#               # 종속변수               # 독립변수
#

