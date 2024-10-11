# tensorflow 모듈 호출
import tensorflow as tf

# 데이터셋 # 손글씨 # mnist 손글씨 이미지 데이터 로드
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
# x_train : 훈련용 28 * 28 픽셀된 0~9 숫자
# y_train : 훈련용 0-9 숫자
# x_valid : 테스트용 28 * 28 픽셀된 0~9 숫자, 독립변수
# y_valid : 테스트용 0-9 숫자, 종속변수
print(x_train.shape, y_train.shape)     # (60000, 28, 28) # 6만개의 28*28 이미지 - 3차원 (60000,) # 1차원
print(x_valid.shape, y_valid.shape)     # (10000, 28, 28) # 1만개의 28*28 이미지 - 3차원 (10000,) # 1차원
# 확인
print(x_train[0, : , : ])   # 첫 번째 데이터의 손글씨 0~255 # 2차원 형식으로 표현
print(y_train[0])           # 첫 번째 데이터의 손글씨 정답 # 5


# 3. 시각화
# 샘플 이미지 출력
import matplotlib.pylab as plt

def plot_image(data, idx) :
    plt.figure(figsize = (5, 5))
    plt.imshow(data[idx], cmap="gray")      # 차트에 이미지 표현 함수 # imshow()
    plt.axis("off")
    plt.show()          # 차트 열기

plot_image(x_train, 0)


# 정규화 전
print(x_train.min(), x_train.max())
print(x_valid.min(), x_valid.max())


# 정규화 # 각 픽셀 값을 255으로 나누어 0 ~ 1 사이의 값으로 변환하여 모델 학습을 더 빠르고 안정적으로 만들기
x_train = x_train / 255.0
x_valid = x_valid / 255.0
# y(종속변수)는 정답이라서 정규화를 하지 않는다

# 정규화 후
print(x_train.min(), x_train.max())
print(x_valid.min(), x_valid.max())


# 채널 추가
# 채널 축 # 채널이란? 색상 정보를 가지는 구성 요소
# 흑백(모노컬러) 이미지를 위해 1개 채널 추가
print(x_train.shape, x_valid.shape)

x_train_in = x_train[..., tf.newaxis]       # 파이썬에서 배열에 축(자원) 추가하는 방법 # ... : 기존 배열 데이터 뜻한다
x_valid_in = x_valid[..., tf.newaxis]       # 3차원 ---> 4차원

print(x_train_in.shape, x_valid_in.shape)

# 모델
# Sequential API를 사용해 샘플 모델 생성
model = tf.keras.Sequential([
    # Convolution 적용(32 filters)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape = (28, 28, 1), name = 'conv'),

    # 풀링 레이어
    # Max Pooling 적용
    tf.keras.layers.MaxPooling2D((2, 2), name='pool'),
    # Classifier 출력층
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax'),
        # 종속변수가  분류할 데이터가 0~9이므로 10개 #다중분류에서는 주로 softmak 활성화 함수를 사용한다
])


# 모델 컴파일 # 옵티마이저, 손실함수, 평가지표 설정
model.compile(optimizer= 'adam', loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])


# 모델 훈련
history = model.fit(x_train_in, y_train,        # 훈련용 데이터와 훈련용 정답
                    validation_data=(x_valid_in, y_valid),      # 테스틍용 데이터와 테스트용 성공
                    epochs=10)


model.evaluate(x_valid_in, y_valid)


# 손실과 정확도 시각화
def plot_loss_acc(history, epoch) :
    loss, val_loss = history.history['loss'], history.history['val_loss']       # 훈련 손실(오차) 값 # 테스트 손실(오차) 값
    acc, val_acc = history.history['accuracy'], history.history['val_accuracy']     # 훈련 정확도 # 테스트 정확도

    # 서브플롯 차트 구성
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(range(1, epoch + 1), loss)
    axes[0].plot(range(1, epoch + 1), val_loss)

    axes[1].plot(range(1, epoch + 1), acc)
    axes[1].plot(range(1, epoch + 1), val_acc)

    plt.show()

plot_loss_acc(history, 10)


# 훈련된 모델로 예측 하기
print(y_valid[0])       # 종속변수 # 10000개 중에 첫 번째 손글씨의 정답 # 숫자
# 7
print(tf.argmax(model.predict(x_valid_in)[0]))     # 독립변수 # 테스트용으로 예측하기
# argmax() : 배열 내 가장 큰 값을 가진 요소의 인덱스 반환
# tf.Tensor(7, shape=(), dtype=int64)


# 모델의 구조
print(model.summary())

# 입력 텐서 형태
print(model.inputs)     # 책에는 model.input 이거지만 model.inputs s붙인 이걸로 하기

# 출력 텐서 형태
print(model.outputs)

# 모델의 전체 레이어
print(model.layers)

# 첫 번째 레이어 선택
print(model.layers[0])

# 첫 번째 레이어 입력 텐서
print(model.layers[0].input)

# 첫 번째 레이어 출력 텐서
print(model.layers[0].output)

# 첫 번째 레이어 가중치
print(model.layers[0].weights)

# 첫 번째 레이어 커널 가중치      # 커널(필터) 행렬의 가중치
print(model.layers[0].kernel)

# 첫 번째 레이어 bias 가중치     # 상수항 # y = ax + b(상수항)
print(model.layers[0].bias)

# 레이어 이름 사용하여 레이어 선택
print(model.gets_layer('conv'))


# 합성곱 시각화 # 합성곱 결과인 특성맵 시각화
activator = tf.keras.Model(inputs = model.inputs, # input  => inputs # 기존 모델의 입력을 사용한다.
                            outputs = layer.ooutput)
# 파이썬 컴프리헨션 : [ f표현식 or 반복변수 in 리스트/range() ]

# 폴링 시각화 # 폴링 결과를 시각화



