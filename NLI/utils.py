import pickle
from collections import Counter
from string import punctuation as puncts

import pandas as pd
import nltk

import torch
from torch.utils.data import Dataset

WORD2INDEX_PATH = "data/winograd_word2index_dict.pkl"

class NLIData(Dataset):

    def __init__(
            self,
            sents_1,
            sents_2,
            labels,
            word2index: dict
        ):
        self.word2index = word2index
        self.entries = []
        max_length_1 = len(max(sents_1, key=len))
        max_length_2 = len(max(sents_2, key=len))
        # print(max_length_1, max_length_2)
        for a, b, label in zip(sents_1, sents_2, labels):
            a_len = len(a)
            if a_len < max_length_1:
                a += [None] * (max_length_1 - a_len)
            b_len = len(b)
            if b_len < max_length_2:
                b += [None] * (max_length_2 - b_len)
            self.entries.append((
                [self.word2index.get(word, self.word2index["<UNK>"]) for word in a],
                a_len,
                [self.word2index.get(word, self.word2index["<UNK>"]) for word in b],
                b_len,
                label
            ))
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, index):
        sent_1_ind, sent_1_len, sent_2_ind, sent_2_len, label = self.entries[index]
        return (
            torch.LongTensor(sent_1_ind), 
            sent_1_len, 
            torch.LongTensor(sent_2_ind), 
            sent_2_len, 
            label
        )

"""
Extracts the sentences and their labels
from the tsv file.
"""
def get_data(path):
    sents_1 = []
    sents_2 = []
    labels = []
    
    df = pd.read_csv(path, sep="\t")
    for index, row in df.iterrows():
        sents_1.append(preprocess_sentence(row["sentence1"]))
        sents_2.append(preprocess_sentence(row["sentence2"]))
        labels.append(row["label"])
    
    return sents_1, sents_2, labels

def preprocess_sentence(sent):
    tokens = nltk.word_tokenize(sent)
    return [word.lower() for word in tokens if word.lower() not in puncts]

"""
Extends an existing vocabulary dictionary or creates a new one if not existing so far.
"""
def extend_vocab(sents, counter=None):
    if counter is None:
        counter = Counter()
    for sent in sents:
        for w in sent:
            counter[w] += 1
    return counter

"""
Creates a word-to-index-dictionary to assign each word a number.

NOTE: Do we need a occurence threshold?
"""
def create_word_index_dict(vocab: dict, occ_treshold=0):
    # keep index zero for padding
    i = 1
    word2index = {None: 0}
    for w, c in vocab.items():
        if c > occ_treshold:
            word2index[w] = i
            i += 1
    word2index['<UNK>'] = i
    
    return word2index

"""
Creates a pickle of the word-to-index-dictionary
to ensure that the model uses the same indices.
"""
def create_word_index_dict_pickle():
    train_sents_1, train_sents_2, _ = get_data("data/train.tsv")
    vocab = extend_vocab(train_sents_1)
    vocab = extend_vocab(train_sents_2, vocab)
    
    val_sents_1, val_sents_2, _ = get_data("data/dev.tsv")
    vocab = extend_vocab(val_sents_1, vocab)
    vocab = extend_vocab(val_sents_2, vocab)
    
    test_sents_1, test_sents_2, _ = get_data("data/test.tsv")
    vocab = extend_vocab(test_sents_1, vocab)
    vocab = extend_vocab(test_sents_2, vocab)
    
    word2index = create_word_index_dict(vocab)
    with open(WORD2INDEX_PATH, 'wb') as fd:
        pickle.dump(word2index, fd, protocol=pickle.HIGHEST_PROTOCOL)

"""
Loads and unpickles the pickled word-to-index-dictionary.
"""
def load_word_index_dict_pickle():
    with open(WORD2INDEX_PATH, 'rb') as fd:
        word2index = pickle.load(fd)
    return word2index

if __name__ == '__main__':
    nltk.download('punkt')
    create_word_index_dict_pickle()
    
    # train_sents_1, train_sents_2, train_labels = get_data("data/train.tsv")
    # vocab = extend_vocab(train_sents_1)
    # vocab = extend_vocab(train_sents_2, vocab)
    
    # val_sents_1, val_sents_2, val_labels = get_data("data/dev.tsv")
    # vocab = extend_vocab(val_sents_1, vocab)
    # vocab = extend_vocab(val_sents_2, vocab)
    
    # test_sents_1, test_sents_2, test_labels = get_data("data/test.tsv")
    # vocab = extend_vocab(test_sents_1, vocab)
    # vocab = extend_vocab(test_sents_2, vocab)

    # word2index = create_word_index_dict(vocab)
    # # print(len(word2index))

    # data = NLIData(train_sents_1, train_sents_2, train_labels, word2index)
    # print(data[0])
