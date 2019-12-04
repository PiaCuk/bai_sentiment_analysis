import numpy as np
import pytreebank
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from bert_embedding import BertEmbedding
from preprocess_utils import filter_minlength

PADDING_LEN = 40

def wordlist_to_bert(wordlist_file):
    x_list = np.load(wordlist_file, allow_pickle=True)
    bert = BertEmbedding(max_seq_length=PADDING_LEN)
    bert_list = []
    for sentence in x_list:
        words, vectors = zip(*bert(sentence))
        flat_vectors = [item for sublist in vectors for item in sublist]
        bert_list.append(np.squeeze(np.asarray(flat_vectors)))
        #print(bert_list[-1].shape)
    return bert_list

def tree_to_bert(tree_labels, verbose=False):
    bert = BertEmbedding(max_seq_length=PADDING_LEN)
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

def embed_bert(partition='train'):
    bert_list = wordlist_to_bert('data/x_'+partition+'_wordlist.npy')
    len_list = [len(i) for i in bert_list]
    print(max(len_list), min(len_list))
    # Pad sequences to same length
    x_padded = pad_sequences(bert_list, maxlen=PADDING_LEN, dtype='float32', padding='post')
    print(x_padded.shape)
    # Save it to .npy files
    np.save('data/x_'+partition+'_new_bert', x_padded)

if __name__ == '__main__':
    embed_bert('dev')
    '''
    preprocess_bert('test')

    Log for bert (entire train dataset)
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