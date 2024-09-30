# p.45

# 1. 텐서플로 모듈 호출
import tensorflow as tf

# 2. 선형 관계를 갖는 데이터 샘플 생성 # Y = 3x - 2
    # 1. 텐서플로의 랜덤 숫자 생성 객체 선언 # 시드값은 아무거나 # 시드란 : 랜덤 생성할 때 사용되는 제어 정수값
g = tf.random.Generator.from_seed(2020)
    # 2. 랜덤 숫자 생성 객체를 이용한 정규분포 난수를 10개 생성해서 벡터(리스트) X에 저장한다.
    # 난수 만들 때 .normal(shape=(축1,)) # .normal(shape=(축1,축2)) # .normal(shape=(축1,축2,축3))
X = g.normal(shape=(10, ))
Y = 3 * X - 2
print(X.numpy())    # 독립 변수 # 피처
# [-0.20943771  1.2746525   1.213214   -0.17576952  1.876984    0.16379918
#   1.082245    0.6199966  -0.44402212  1.3048344 ]
print(Y.numpy())    # 종속 변수 # 타겟
# [-2.628313    1.8239574   1.6396422  -2.5273085   3.630952   -1.5086024
#   1.2467351  -0.14001012 -3.3320663   1.9145031 ]

# 3. Loss 함수 정의 # 손실 함수(평균 제곱 오차)를 정의하는 함수
def cal_msg(X, Y, a, b) :   # Y : 실제값
    Y_pred = a * X + b  # Y값-종속(예측) = 계수(기울기)a * X(피처) + 상수항(Y절편) # ((피처값을 넣어서 Y값 종속값을 예측한다))
    squaared_error = (Y_pred - Y) ** 2   # 예측 Y와 실제 Y 간의 차이의 제곱을 계산(오차 제곱)
    mean_squared_error = tf.reduce_mean(squaared_error)     # 모든 오차 제곱의 평균을 계산 하여 반환
    print(mean_squared_error)
    return mean_squared_error

# 4. 자동 미분 과정을 기록
a = tf.Variable(0.0)    # 계수 # 텐서플로 변수에 0으로 초기화 # 기울기
b = tf.Variable(0.0)    # y절편 # 텐서플로 변수에 0.0으로 초기화
# 목적 : a와 b를 미세하게 변경하면서 반복적으로 계산하여 손실을 최소화 하는 값을 찾는다.

EPOCHS = 200    # 훈련 횟수 # 에포크

for epoch in range(1, EPOCHS+1) : # 1 ~ 200까지 (200회)
    # 200번을 반복하면서 목적 : a와 b를 미세하게 변경하면서 차이가 가장 적은 값을 찾자.

    # 4-1 # msg 기록? # tf.GradientTape() as 변수 : with 안에 있는 계산식들을 모두 기록하는 역할 # mse를 tape에 기록한다.
    with tf.GradientTape() as tape :
        mse = cal_msg(X, Y, a, b) # 위에서 정의한 손실함수를 계산한다

    # 4-2 기울기 계산 # tape.gradient()를 이용하여 mse에 대한 a와 b의 미분값(기울기)을 구한다.
    grad = tape.gradient(mse, {'a' : a, 'b' : b})  # mse에 대한 a와 b를 딕셔너리 반환한다.
    d_a = grad['a']
    d_b = grad['b']

    # 4-3 # .assign_sub() 텐서플로 변수에 매개변수를 원본값에서 뺀 값으로 변수값을 수정하는 함수
    a.assign_sub(d_a * 0.05)    # 현재값의 5% 감소
    b.assign_sub(d_b * 0.05)    # 0.05 감소

    # 4-4 # 중간 계산 확인
    if epoch % 20 == 0 :    # 20번마다 # epoch=반복횟수 # mse : 평균제곱오차 # a계수 # b상수항
        print(f'{epoch}, {mse:4f}, {a.numpy():4f}, {b.numpy():4f}')

