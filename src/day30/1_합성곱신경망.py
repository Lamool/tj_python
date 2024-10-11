import tensorflow as tf

# 1. 데이터셋 로드, 10가지 종류의 의류 이미지 데이터셋
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train) , (x_valid, y_valid) = fashion_mnist.load_data()

# 2. 데이터 시각화
import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()

# 위 데이터셋을 이용한 합성곱 모델 구축하고 학습하여 정확도(accuracy) 95% 이상 되도록 최적의 하이퍼 파라미터 설정 하여 모델 만들기

print(x_train.min(), x_train.max())
print(x_valid.min(), x_valid.max())

# 정규화
x_train = x_train / 255.0
x_valid = x_valid / 255.0
print(x_train.min(), x_train.max())
print(x_valid.min(), x_valid.max())

# 채널 추가
print (x_train.shape, x_valid.shape)

x_train_in = x_train[..., tf.newaxis]
x_valid_in = x_valid[..., tf.newaxis]

print(x_train_in.shape, x_valid_in.shape)

# Sequential  API를 사용해 샘플 모델 생성
model = tf.keras.Sequential([
    tf.keras.oayers.Conv2D
])

# 모델 훈련
# model.fit(x_tra


# 최적의 파라미터 찾기 위해서는 1. epochs 조정 2.



