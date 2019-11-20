import numpy as np
from keras.models import Sequential
from keras.layers import Input, Convolution1D, Dense, Flatten, Dropout
from keras.utils import to_categorical

_SEQ_SHAPE = (25, 768)

x_train = np.load('data/x_train_bert.npy')
y_train = np.load('data/y_train_bert.npy')
# This should be done with the data preprocessing
# y_train = to_categorical(y_ints, num_classes=5)
print(y_train.shape)
print(x_train.shape)
x_val = np.load('data/x_dev_bert.npy')
y_val = np.load('data/y_dev_bert.npy')

model = Sequential()
# Convolutional model (3x conv, flatten, 2x dense)
model.add(Convolution1D(64, 3, padding='same', input_shape=_SEQ_SHAPE))
model.add(Convolution1D(32, 3, padding='same'))
model.add(Convolution1D(16, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(64 ,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# TODO implement callbacks savebestmodel, tensorboard
out = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_val, y_val))

scores = model.evaluate(x_val, y_val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Accuracy: 51.71% for bert 3 epochs