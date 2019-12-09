import numpy as np
from keras.preprocessing.sequence import pad_sequences

def load_glove_embedding(directory = 'data/glove.840B.300d.txt'):
    print('Indexing word vectors.')
    embeddings_index = {}

    # load glove embedding to dict
    with open(directory) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def tree_to_glove(train_set, embeddings_index):
    x_list = []
    x_array = []

    for sentence in train_set:
        for word in sentence:
            if embeddings_index.get(str(word)) is not None:
                if len(embeddings_index.get(str(word))):
                    x = embeddings_index.get(str(word))
                    x_list.append(x)
        x_array.append(x_list)
        x_list = []
    return x_array


def preprocess_glove(partition='train'):
    print("Beginn")

    #load glove
    embedded_list = load_glove_embedding('data/glove.840B.300d.txt') 
    # load train_set
    train_set = np.load('data/x_'+ str(partition) +'_wordlist.npy', allow_pickle=True)

    x_list = tree_to_glove(train_set, embedded_list)

    print("All "+partition+" samples, w/o filtering " + str(len(x_list)))

    # Pad sequences to same length
    # Max length for the training dataset is 23, so padding to 25 to make sure
    x_padded = pad_sequences(x_list, maxlen=40, dtype='float32', padding='post')
    print(x_padded.shape)

    # Save it all to .npy files
    np.save('data/x_'+partition+'_glove', x_padded)
 

preprocess_glove('train')
