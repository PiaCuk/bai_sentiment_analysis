import numpy as np
import pytreebank
from keras_preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from bert_embedding import BertEmbedding
from allennlp.commands.elmo import ElmoEmbedder


def tree_to_elmo(tree_labels, verbose=False):
    #bert = BertEmbedding()
    elmo = ElmoEmbedder()
    #y_list = []
    elmo_list = []
    for t in tree_labels:
        #y, x = t.to_labeled_lines()[0]
        #y_list.append(y)
        # str_list, arr_list = zip(*bert([x]))
        # if verbose:
        #    print()
        elmo_vec = elmo.embed_sentence(t)
        elmo_list.append(np.asarray(elmo_vec))
        #print(elmo_list[-1].shape)
        #print(x_list[-1].shape)

    return elmo_list

def filter_minlength_elmo(sequences, labels, min_length=5):
    y_x_filtered = [z for z in zip(labels, sequences) if z[1].shape[1] > min_length]
    #print(y_x_filtered)
    filtered_labels, filtered_sequences = zip(*y_x_filtered)
    return filtered_sequences, filtered_labels


def preprocess_elmo(partition='train'):
    #sst_data = pytreebank.load_sst()
    # Load the dataset and vectorize it
    #train_set = sst_data[partition]
    print("Beginn")
    train_set = np.load('data/x_test_wordlist.npy', allow_pickle=True)
    x_list = tree_to_elmo(train_set)
    print("All "+partition+" samples, w/o filtering " + str(len(x_list)))

    # Filter for min sentence length
    # Setting the min length to 5, this is the 3rd percentile
    # Also, it makes sense for the sentences to be at least 5 words
    #x_filtered, y_filtered = filter_minlength_elmo(x_list, y_list, min_length=5)
    #print("All "+partition+" samples, w/ filtering " + str(len(x_filtered)))

    #xfiltered = []
    xlist = []
    for n in x_list:
        xlist.append(n.reshape(np.ma.size(n, 1) * 3, 1024))
        #print(n.shape)

    # Pad sequences to same length
    # Max length for the training dataset is 23, so padding to 25 to make sure
    x_padded = pad_sequences(xlist, maxlen=40, dtype='float32', padding='post')
    print(x_padded.shape)

    # One-hot encode labels. This is necessary to use loss=categorical_crossentropy for training
    #y_onehot = to_categorical(y_filtered, num_classes=5)

    # Save it all to .npy files
    np.save('data/x_'+partition+'_elmo', x_padded)
    #np.save('data/y_'+partition+'_elmo1', y_onehot)


preprocess_elmo('test')
'''Log for bert (entire train dataset)
Using TensorFlow backend.
All training samples, w/o filtering 8544
All training samples, w/ filtering 8177
(8177, 25, 768)

Log for bert (dev)
All dev samples, w/o filtering 1101
All dev samples, w/ filtering 1068
(1068, 25, 768)

Log for bert (test)
All test samples, w/o filtering 2210
All test samples, w/ filtering 2149
(2149, 25, 768)
'''