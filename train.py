import numpy as np
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
from models import conv_model, keras_conv_model, lstm_model

# elmo: 
#_SEQ_SHAPE = (25, 1024)
# bert:
#_SEQ_SHAPE = (25, 768)
# Keras embedding:
_SEQ_SHAPE = 40

# decided on CNN 256 and LSTM 128-64 as our two test architectures
# specify embedding in test() and train() with parameter "embedding", default = bert
def main():
    #train(embedding='keras')
    test('models/weights.01-1.59.hdf5', embedding = 'keras')
    '''
    BERT: 
    Accuracy: 46.21% for keras_128    
    Accuracy: 47.18% for keras_256    
    Accuracy: 46.07% for keras_512    
    Accuracy: 47.56% for keras_512-256-128    
    Accuracy: 47.51% for lstm_64 
    Accuracy: 48.30% for lstm_128
    Accuracy: 48.67% for lstm_128-64
    Accuracy: 45.84% for conv_model

    ELMO:
    Accuracy: 44,95% for keras_128 
    Accuracy: 46,07% for keras_256 
    Accuracy: 45.46% for keras_512
    Accuracy: 44.63% for keras_512-256-128
    Accuracy: 44.81% for lstm_64
    Accuracy: 44.63% for lstm_128
    Accuracy: 44.21% for lstm_128-64
    Accuracy: 44.21% for conv_model

    Keras:
    Accuracy: 27.98% for keras_128
    Accuracy: 25.35% for keras_256
    '''

###############################################################################################################
def train(plot=True, embedding = 'bert'):
    x_train = np.load('data/x_train_' + embedding + '.npy', allow_pickle=True)
    y_train = np.load('data/y_train_' + embedding + '.npy', allow_pickle=True)
    x_val = np.load('data/x_dev_' + embedding + '.npy', allow_pickle=True)
    y_val = np.load('data/y_dev_' + embedding + '.npy', allow_pickle=True)
    print(y_train.shape)
    print(x_train.shape)

    model = keras_conv_model(_SEQ_SHAPE)
    

    save_best_model = ModelCheckpoint('models/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, period=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=0)
    tensorboard = TensorBoard(log_dir='logs', write_graph=True)

    out = model.fit(x_train, y_train, epochs=20, callbacks=[save_best_model, early_stopping, tensorboard],
                    batch_size=64, validation_data=(x_val, y_val))
    if plot:
        # Summarize history for loss
        plt.figure()
        plt.plot(out.history['loss'])
        plt.plot(out.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

def test(ckpt, embedding = 'elmo'):
    x_test = np.load('data/x_test_' + embedding + '.npy')
    y_test = np.load('data/y_test_' + embedding + '.npy')

    model = keras_conv_model(_SEQ_SHAPE, load_weights=ckpt)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

###############################################################################################################
if __name__ == '__main__':
    main()