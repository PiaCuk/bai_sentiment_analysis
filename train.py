import numpy as np
from keras.models import Sequential
from keras.layers import Input, Convolution1D, Dense, Flatten, Dropout
from keras.utils import to_categorical

_SEQ_SHAPE = (25, 768)

x_train = np.load('x_train_bert500.npy')
y_train = np.load('y_train_bert500.npy')
# This should be done with the data preprocessing
# y_train = to_categorical(y_ints, num_classes=5)

model = Sequential()
# Convolutional model (3x conv, flatten, 2x dense)
model.add(Convolution1D(64, 3, padding='same', input_shape=_SEQ_SHAPE))
model.add(Convolution1D(32, 3, padding='same'))
model.add(Convolution1D(16, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(32 ,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, epochs=10, batch_size=64)

scores = model.evaluate(x_train, y_train, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Accuracy: 66.87% for bert500