import numpy as np
import glob
import os
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from models import conv_model, keras_conv_model, lstm_model

# elmo: 
#_SEQ_SHAPE = (40, 1024)
# bert:
_SEQ_SHAPE = (40, 768) # Used to be 25 w/ BertEmbedding(max_seq_legth=25) default
# Keras embedding:
#_SEQ_SHAPE = 40
# glove embedding 300:
#_SEQ_SHAPE = (40, 300)
# glove embedding small:
#_SEQ_SHAPE = (40, 50)
EMBEDDING = 'new_bert'

# decided on CNN 256 and LSTM 128-64 as our two test architectures
# specify embedding in test() and train() with parameter "embedding", default = bert
def main():
    '''
    log_file = open('models/'+EMBEDDING+'_logfile.txt', 'a')
    score_list = []
    for trial in range(3):
        train(embedding=EMBEDDING, trial=EMBEDDING+str(trial), verbose=False)
        scores = test(embedding=EMBEDDING)
        accuracy = scores[1]*100
        print("Accuracy: %.2f%%" % (accuracy))
        score_list.append(accuracy)
        log_file.write("%.3f\n" % accuracy)
    
    print("Averaged accuracy: %.2f%%" % (np.mean(score_list)))
    log_file.close()
    '''
    class_report(embedding=EMBEDDING, ckpt='models/new_bert1_weights.10-1.34.hdf5')

###############################################################################################################
def train(embedding='bert', trial='trial', verbose=False, plot=False):
    x_train = np.load('data/x_train_' + embedding + '.npy', allow_pickle=True)
    y_train = np.load('data/y_train_' + embedding + '.npy', allow_pickle=True)
    x_val = np.load('data/x_dev_' + embedding + '.npy', allow_pickle=True)
    y_val = np.load('data/y_dev_' + embedding + '.npy', allow_pickle=True)
    if verbose:
        print(y_train.shape)
        print(x_train.shape)

    model = lstm_model(_SEQ_SHAPE, verbose=verbose)
    
    save_best_model = ModelCheckpoint('models/'+trial+'_weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                    monitor='val_loss', verbose=0, save_best_only=True, period=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=0)
    #tensorboard = TensorBoard(log_dir='logs', write_graph=True)

    out = model.fit(x_train, y_train, epochs=20, callbacks=[save_best_model, early_stopping], #, tensorboard],
                    batch_size=64, validation_data=(x_val, y_val), verbose=2 if verbose==True else 0)
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

def test(embedding='bert', ckpt=None, verbose=False):
    if ckpt is not None:
        latest_ckpt = ckpt
    else:
        list_of_files = glob.glob('models/*.hdf5')
        latest_ckpt = max(list_of_files, key=os.path.getatime)
    
    x_test = np.load('data/x_test_' + embedding + '.npy')
    y_test = np.load('data/y_test_' + embedding + '.npy')

    model = lstm_model(_SEQ_SHAPE, load_weights=latest_ckpt, verbose=verbose)

    scores = model.evaluate(x_test, y_test, verbose=0)
    return scores

def class_report(embedding='bert', ckpt=None, verbose=False):
    if ckpt is not None:
        latest_ckpt = ckpt
    else:
        list_of_files = glob.glob('models/*.hdf5')
        latest_ckpt = max(list_of_files, key=os.path.getatime)
        
    x_test = np.load('data/x_test_' + embedding + '.npy')
    y_test = np.load('data/y_test_' + embedding + '.npy')

    model = lstm_model(_SEQ_SHAPE, load_weights=latest_ckpt, verbose=verbose)
        
    # Confution Matrix and Classification Report
    Y_pred = model.predict(x_test, batch_size=64)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    #print(y_true)
    print('Confusion Matrix')
    print(confusion_matrix(y_true, y_pred, labels=[*range(5)], normalize='true'))
    print('Classification Report')
    print(classification_report(y_true, y_pred, labels=[*range(5)]))    

###############################################################################################################
if __name__ == '__main__':
    main()
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