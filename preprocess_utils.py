from keras_preprocessing.text import text_to_word_sequence


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