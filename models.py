from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout

def conv_model(input_shape, drop_rate=0.5, load_weights=None, verbose=True):
    model = Sequential()
    # Convolutional model (3x conv, flatten, 2x dense)
    model.add(Conv1D(64, 3, padding='same', input_shape=input_shape))
    model.add(Conv1D(32, 3, padding='same'))
    model.add(Conv1D(16, 3, padding='same'))
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

def keras_conv_model(input_shape, load_weights=None, verbose=True):
    model = Sequential()
    
    model.add(Conv1D(512, 3, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling1D(3, strides=2))
    model.add(Conv1D(512, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(3, strides=2))
    model.add(Conv1D(512, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(5))  # global max pooling
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    
    if load_weights is not None:
        model.load_weights(load_weights)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if verbose:
        print(model.summary())
    return model

def lstm_model(input_shape, drop_rate=0.5, load_weights=None, verbose=True):
    model = Sequential()
    model.add(LSTM(input_shape=input_shape, units=128, return_sequences=True))
    model.add(Dropout(drop_rate))
    model.add(LSTM(units=64))
    model.add(Dropout(drop_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(5, activation='softmax'))
    
    if load_weights is not None:
        model.load_weights(load_weights)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if verbose:
        print(model.summary())
    return model