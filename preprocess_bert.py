import numpy as np
import pytreebank
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from bert_embedding import BertEmbedding
from preprocess_utils import tree_to_wordlist


def wordlist_to_bert(labeled_tree):
    x_list, y_list = tree_to_wordlist(labeled_tree)
    bert = BertEmbedding()
    bert_list = []
    for sentence in x_list:
        words, vectors = zip(*bert(sentence))
        print(words)
        bert_list.append(np.squeeze(np.asarray(vectors)))
    return np.asarray(bert_list), y_list

def tree_to_bert(tree_labels, verbose=False):
    bert = BertEmbedding()
    x_list = []
    y_list = []
    for t in tree_labels:
        y, x = t.to_labeled_lines()[0]
        y_list.append(y)
        str_list, arr_list = zip(*bert([x]))
        if verbose:
            print(str_list)
        x_list.append(np.squeeze(np.asarray(arr_list)))
        # print(x_list[-1].shape)
    return x_list, y_list

def filter_minlength(sequences, labels, min_length=5):
    y_x_filtered = [z for z in zip(labels, sequences) if z[1].shape[0] > min_length]
    filtered_labels, filtered_sequences = zip(*y_x_filtered)
    return filtered_sequences, filtered_labels

def preprocess_bert(partition='train'):
    sst_data = pytreebank.load_sst()
    # Load the dataset and vectorize it
    train_set = sst_data[partition]
    x_list, y_list = tree_to_bert(train_set)
    print("All "+partition+" samples, w/o filtering " + str(len(x_list)))

    # Filter for min sentence length
    # Setting the min length to 5, this is the 3rd percentile
    # Also, it makes sense for the sentences to be at least 5 words
    x_filtered, y_filtered = filter_minlength(x_list, y_list, min_length=5)
    print("All "+partition+" samples, w/ filtering " + str(len(x_filtered)))

    # Pad sequences to same length
    # Max length for the training dataset is 23, so padding to 25 to make sure
    x_padded = pad_sequences(x_filtered, maxlen=25, dtype='float32', padding='post')
    print(x_padded.shape)

    # One-hot encode labels. This is necessary to use loss=categorical_crossentropy for training
    y_onehot = to_categorical(y_filtered, num_classes=5)

    # Save it all to .npy files
    np.save('data/x_'+partition+'_bert', x_padded)
    np.save('data/y_'+partition+'_bert', y_onehot)


preprocess_bert('test')
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