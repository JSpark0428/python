import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

# MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 입력 데이터 재구성 및 정규화
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train/255.0
x_test = x_test/255.0

# 레이블을 범주형 형식으로 변환
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 모델 구성
# 히든 레이어 수를 늘리거나 줄여봤으나, 큰 차이는 없ㅋ
n_input = 784
n_hidden1 = 1024
n_hidden2 = 512
n_hidden3 = 512
n_hidden4 = 512
#n_hidden5 = 512
#n_hidden6 = 512
n_output = 10

mlp = Sequential()
mlp.add(Dense(units = n_hidden1, activation = 'tanh',
              input_shape=(n_input,), kernel_initializer = 'random_uniform', bias_initializer ='zeros'))
mlp.add(Dense(units = n_hidden2, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))
mlp.add(Dense(units = n_hidden3, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer = 'zeros')) 
mlp.add(Dense(units = n_hidden4, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))
#mlp.add(Dense(units = n_hidden5, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer = 'zeros')) 
#mlp.add(Dense(units = n_hidden6, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))
mlp.add(Dense(units = n_output, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))

# 모델 컴파일
mlp.compile(loss = 'mse', optimizer = Adam(learning_rate=0.001), metrics = ['accuracy'])
hist = mlp.fit(x_train, y_train, batch_size = 128, epochs = 30, validation_data = (x_test, y_test), verbose =2)

# 모델 훈련
res = mlp.evaluate(x_test, y_test, verbose = 0)
print("Accuracy is", res[1]*100)

# 정확도 그래프
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'lower right')
plt.grid()
plt.show()

#손실 그래프
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc = 'upper right')
plt.grid()
plt.show()