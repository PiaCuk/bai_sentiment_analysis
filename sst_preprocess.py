# TODO transfer ipynb to py
import numpy as np
import pytreebank
from keras_preprocessing.text import text_to_word_sequence
from bert_embedding import BertEmbedding
from data_utils import rpad


def tree_to_wordlist(tree_labels):
    x_list = []
    y_list = []
    for t in tree_labels:
        y, x = t.to_labeled_lines()[0]
        y_list.append(y)
        x = text_to_word_sequence(x, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`´{|}~\t\n'+"'")
        x_list.append(x)
    print(len(x_list))
    print(len(y_list))
    return (x_list, y_list)

def wordlist_to_bert(wordlist):
    bert = BertEmbedding()
    bert_list = []
    for sentence in wordlist:
        words, vectors = zip(*bert(sentence))
        print(words)
        bert_list.append(np.squeeze(np.asarray(vectors)))
    return np.asarray(bert_list)

def tree_to_bert(tree_labels):
    bert = BertEmbedding()
    x_list = []
    y_list = []
    for t in tree_labels:
        y, x = t.to_labeled_lines()[0]
        y_list.append(y)
        str_list, arr_list = zip(*bert([x]))
        print(str_list)
        arr_list = rpad(arr_list, 30)
        x_list.append(np.asarray(arr_list))
        print(x_list[-1].shape)
    return x_list, y_list

sst_data = pytreebank.load_sst()
"""
train_set = sst_data['train'][:1]
print(len(train_set))
x_train, y_train = tree_to_wordlist(train_set)
x_train = wordlist_to_bert(x_train)
np.save('test', x_train)
"""
train_set = sst_data['train'][:3]
x_train, y_train = tree_to_bert(train_set)
# x_train = np.asarray(x_train)
print(x_train.shape)
print(x_train[0].shape)