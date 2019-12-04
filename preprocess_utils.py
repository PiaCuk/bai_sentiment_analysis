import numpy as np
import re
import pytreebank
from keras_preprocessing.text import text_to_word_sequence, Tokenizer
from bert_embedding import BertEmbedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def filter_minlength(sequences, labels, min_length=5):
    y_x_filtered = [z for z in zip(labels, sequences) if z[1].shape[0] > min_length]
    filtered_labels, filtered_sequences = zip(*y_x_filtered)
    
    return filtered_sequences, filtered_labels


def tree_to_wordlist(tree_labels, use_tokenizer=False, bert_filtering=False):
    x_list = []
    y_list = []
    
    for t in tree_labels:
        y, x = t.to_labeled_lines()[0]
        y_list.append(y)
        if bert_filtering:
            bert = BertEmbedding()
            x, embedded_x = zip(*bert([x]))
        else:
            x = text_to_word_sequence(x)
        x_list.append(x)
    
    print(len(x_list))
    y_x_filtered = [z for z in zip(y_list, x_list) if len(z[1]) > 2 and len(z[1]) < 40]
    y_list, x_list = zip(*y_x_filtered)
    
    if use_tokenizer:
        tokenizer = Tokenizer(num_words=1000)
        tokenizer.fit_on_texts(x_list)
        x_list = tokenizer.texts_to_sequences(x_list)
    
    return (x_list, y_list)

def tokenize(partition='train'):
    sst_data = pytreebank.load_sst()
    # Load the dataset and vectorize it
    train_set = sst_data[partition]
    x_list, y_list = tree_to_wordlist(train_set, use_tokenizer=True)
    print("All "+partition+" samples, w/ filtering " + str(len(x_list)))
    
    # Convert to numpy array
    x_array = [np.asarray(x) for x in x_list]
    x_array = pad_sequences(x_array, maxlen=40, dtype='int32', padding='post')
    print(x_array.shape)

    # One-hot encode labels. This is necessary to use loss=categorical_crossentropy for training
    y_array = to_categorical(y_list, num_classes=5)
    print(y_array.shape)

    # Save it all to .npy files
    np.save('data/x_'+partition+'_keras', x_array)
    np.save('data/y_'+partition+'_keras', y_array)



if __name__ == '__main__':
    tokenize('test')
    ''' Log
    8544
    All train samples, w/ filtering 8302
    1101
    All dev samples, w/ filtering 1085
    2210
    All test samples, w/ filtering 2166
    '''