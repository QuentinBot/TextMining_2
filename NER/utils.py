import string
import pickle
import torch

from collections import Counter
from torch.utils.data import Dataset

WORD2INDEX_PATH = "data/conll2003word2index_dict.pkl"

class NERData(Dataset):

    """
    Creates a dataset of sentences and tags for the torch Data loader.

    NOTE: Do we need a sentence length threshold?
    """
    def __init__(self, sents, tags, word2index, sent_length_threshold=0):
        self.entries = []
        for sent, sent_tags in zip(sents, tags):
            entry_word_ind = []
            entry_tags = []
            for word, tag in zip(sent, sent_tags):
                if word in word2index:
                    entry_word_ind.append(word)
                    entry_tags.append(tag)
            if len(entry_word_ind) > sent_length_threshold:
                self.entries.append((entry_word_ind, entry_tags))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        sent_ind, sent_tags = self.entries[index]
        return torch.LongTensor(sent_ind), torch.LongTensor(sent_tags)

def get_data(path):
    sents = []
    sents_tags = []
    
    with open(path, 'rb') as fd:
        data = pickle.load(fd)
    
        for sent, sent_tags in data:
            sents.append(sent)
            sents_tags.append(sent_tags)
    
    return sents, sents_tags

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
    i = 1
    word2index = {}
    for w, c in vocab.items():
        if c > occ_treshold:
            word2index[w] = i
            i += 1
    
    return word2index

"""
Creates a pickle of the word-to-index-dictionary
to ensure that the model uses the same indices.
"""
def create_word_index_dict_pickle():
    sents, _ = get_data("data/conll2003_train.pkl")
    vocab = extend_vocab(sents)
    sents, _ = get_data("data/conll2003_val.pkl")
    vocab = extend_vocab(sents, vocab)
    sents, _ = get_data("data/conll2003_test.pkl")
    vocab = extend_vocab(sents, vocab)
    word2index = create_word_index_dict(vocab)
    with open(WORD2INDEX_PATH, 'wb') as fd:
        pickle.dump(word2index, fd, protocol=pickle.HIGHEST_PROTOCOL)

def load_word_index_dict_pickle():
    with open(WORD2INDEX_PATH, 'rb') as fd:
        word2index = pickle.load(fd)
    return word2index

# for debug purposes
if __name__ == '__main__':
    create_word_index_dict_pickle()
    # print(len(load_word_index_dict_pickle()))
