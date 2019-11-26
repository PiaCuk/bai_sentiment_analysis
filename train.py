import numpy as np
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
from models import conv_model, keras_conv_model, lstm_model

# elmo: 
#_SEQ_SHAPE = (25, 1024)
# bert:
_SEQ_SHAPE = (25, 768)

# specify embedding in test() and train() with parameter "embedding", default = bert
def main():
    #train(embedding='bert')
    test('models/weights.04-1.23.hdf5', embedding = 'bert')
    '''
    Accuracy: 46.21% for keras_128 and bert embedding
    Accuracy: 47.18% for keras_256 and bert embedding
    Accuracy: 46.07% for keras_512 and bert embedding
    Accuracy: 47.56% for keras_512-256-128 and bert embedding
    Accuracy: 45.84% for conv_model and bert
    Accuracy: 48.30% for lstm_128 and bert embedding
    Accuracy: 47.51% for lstm_64 and bert embedding
    Accuracy: 48.67% for lstm_128-64 and bert embedding
    '''

###############################################################################################################
def train(plot=True, embedding = 'bert'):
    x_train = np.load('data/x_train_' + embedding + '.npy')
    y_train = np.load('data/y_train_' + embedding + '.npy')
    x_val = np.load('data/x_dev_' + embedding + '.npy')
    y_val = np.load('data/y_dev_' + embedding + '.npy')

    model = lstm_model(_SEQ_SHAPE)
    print(y_train.shape)
    print(x_train.shape)

    save_best_model = ModelCheckpoint('models/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, period=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, verbose=0)
    tensorboard = TensorBoard(log_dir='logs', write_graph=True)

    out = model.fit(x_train, y_train, epochs=5, callbacks=[save_best_model, early_stopping, tensorboard],
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

    model = lstm_model(_SEQ_SHAPE, load_weights=ckpt)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

###############################################################################################################
if __name__ == '__main__':
    main()