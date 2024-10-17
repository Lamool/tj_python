# 실무에서는 데이터셋을 웹에서 로드하는 경우가 거의 없다
# 직접 이미지를 로드하는 경우가 더 많다

import tensorflow as tf
import zipfile
import os

# 1. 데이터 준비
    # 데이터 경로 위치
source_filename = 'c:/doit/dataset/cat-and-dog.zip'        # (1) zip파일이 위치한 파일명
extract_folder = 'c:/doit/dataset'         # (2) zip파일을 압축해제할 폴더명
    # (파이썬 코드로) 압축 해제
with zipfile.ZipFile(source_filename, 'r') as zipObj:
    zipObj.extractall(extract_folder)      # 지정한 경로에 압축해제 하기
        # zipfile.ZipFile(source_filename, 'r') : zip파일을 읽기모드를 읽어와서 zipObj 변수에 담기
        # 파일객체변수명.extractall(압축해제할폴더경로)
    # 훈련용, 검증용 저장 위치 지정
train_dir = os.path.join(extract_folder, "archive/training_set/training_set")
valid_dir = os.path.join(extract_folder, "archive/test_set/test_set")
print(train_dir)
print(valid_dir)

# 2. 정규화
from tensorflow.keras.preprocessing.image import ImageDataGenerator     # 모듈 호출
image_gen = ImageDataGenerator(rescale=(1/255.0))        # 이미지 데이터
# 3. 이미지 제네레이터 : 한 번에 많은 데이터를 가져오면 메모리에 문제가 살행 하므로 이미지를 배치(묶음) 단위로 반복해서 가져오기
    # flow_from_directory()
train_gen = image_gen.flow_from_directory(train_dir,
                                          batch_size=16,                # batch_size를 줄여보는 방법이 있다. 돌아가다 꺼진 경우
                                          target_size=(244, 244),
                                          classes = ['cats', 'dogs'],
                                          class_mode = 'binary',
                                          seed = 2020
                                          )

valid_gen = image_gen.flow_from_directory(valid_dir,
                                          batch_size=16,
                                          target_size=(244, 244),
                                          classes = ['cats', 'dogs'],
                                          class_mode = 'binary',
                                          seed = 2020
                                          )
# 4. 샘플 데이터 시각화
class_labels = ['cats', 'dogs']
batch = next(train_gen)
images =batch[0]
labels = batch[1]

import matplotlib.pyplot as plt
for i in range(16) :
    ex = plt.subplot(4,8, i +1)
    plt.imshow(images[i])
    plt.title(class_labels[int(labels[i])])     # int로 변환
plt.show()


# Sequential API를 사용하여 샘플 모델 생성
def build_model() :
    model = tf.keras.Sequential([
        # Convolution층
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Classifier 출력층
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    return model

model = build_model()

# 모델 컴파일
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 모델 훈련
history = model.fit(train_gen, validation_data=valid_gen, epochs=5)

# 손실함수, 정확도 그래프 그리기
# plot_loss_acc(history, 5)


