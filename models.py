from keras.models import Sequential
from keras.layers import Input, Convolution1D, Dense, Flatten, Dropout

def conv_model(input_shape, drop_rate=0.5, load_weights=None, verbose=True):
    model = Sequential()
    # Convolutional model (3x conv, flatten, 2x dense)
    model.add(Convolution1D(64, 3, padding='same', input_shape=input_shape))
    model.add(Convolution1D(32, 3, padding='same'))
    model.add(Convolution1D(16, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(drop_rate))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(drop_rate))
    model.add(Dense(5, activation='softmax'))

    if load_weights is not None:
        model.load_weights(load_weights)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if verbose:
        print(model.summary())
    return model