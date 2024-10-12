import tensorflow as tf

# 데이터셋 # 10가지의 종류의 이미지 데이터셋 [ 비행기, 자동차, 새, 고양이, 개구리, 사슴, 개 말, 배, 트럭 ]
#cifar10 = tf.keras.datasets.cifar10

# 칼라 이미지의 합성곱 모델 만들기
(x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.cifar10.load_data()

# 확인용
# print(x_train.shape, y_train.shape)
# print(x_valid.shape, y_valid.shape)

# 2. 데이터 시각화
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0])
# plt.show()

# 칼라 이미지의 합성곱 모델 만들기
# 정규화
x_train = x_train / 250.0
x_valid = x_valid / 255.0

# Sequential API를 사용해 샘플 모델 생성
# 채널 추가
# x_train_in = x_train[..., tf.newaxis]       # 데이터셋 자체에 이미 돼있어서(?) 안 해도 된다
# x_valid_in = x_valid[..., tf.newaxis]

# 모델
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=10)


#################### 예측
# 1. 파이썬 OpenCV : 이미지 파일을 파이썬으로 호출 하는 모듈 제공한다.
import cv2      # opencv-python 설치

# 2. 외부 이미지 가져오기
img = cv2.imread('dog.png')
print(img)
print(img.shape)        # (336, 575, 3) : 원본 이미지는 가로 336픽셀 세로 575픽셀 칼라(3채널)

# 3. 이미지의 사이즈 변경
img = cv2.resize(img, dsize=(32, 32))       # 모델이 학습한 사이즈와 동일하게 변경  # (32, 32, 3) 픽셀 줄이기
print(img.shape)

# 정규화
img = img / 255.0

# 4. 변경된 이미지 cv 시각화
cv2.imshow('img', img)
cv2.waitKey()

# 5. 모델을 이용한 새로운 이미지 예측하기
result = model.predict(img[tf.newaxis, ...])     # (32, 32, 3) --> (1, 32, 32, 3)
print(tf.argmax(result[0]).numpy())     # 가장 높은 확률을 가진 종속변수

# 1. 정규화 안 했더니 예측값 - 8
# 2. ((정확도 올리기))
# 3. 레이어 추가했더니 예측값 5

########################## 외부 '자동차' 이미지의 예측
img = cv2.imread('car.png')
img = cv2.resize(img , dsize=( 32 , 32))
img = img / 255.0
result = model.predict(img[tf.newaxis, ... ])     # ( 32 , 32 , 3 ) --> ( 1 , 32 , 32 , 3 )
print(tf.argmax(result[0]).numpy())              # 가장 높은 확률을 가진 종속변수 # 1

