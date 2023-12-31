import pickle
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

WORD2INDEX_PATH = "data/conll2003word2index_dict.pkl"

class NERData(Dataset):

    """
    Creates a dataset of sentences and tags for the torch Data loader.
    """
    def __init__(
            self,
            sents,
            tags, 
            word2index: dict,
        ):
        self.entries = []
        for sent, sent_tags in zip(sents, tags):
            entry_word_ind = []
            entry_tags = []
            for word, tag in zip(sent, sent_tags):
                entry_word_ind.append(word2index.get(word, 0))
                entry_tags.append(tag)
            self.entries.append((entry_word_ind, entry_tags))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        sent_ind, sent_tags = self.entries[index]
        return torch.LongTensor(sent_ind), torch.LongTensor(sent_tags)

"""
Extracts the sentences and its corresponding tag sequences
from the pickled data.
"""
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
    # keep index zero for padding
    i = 1
    word2index = {None: 0}
    for w, c in vocab.items():
        if c > occ_treshold:
            word2index[w] = i
            i += 1
    
    return word2index

"""
Splits every sentence and its corresponding tag sequence
into windows of fixed size such that the torch DataLoader
can iterate over batches of similar sizes.

In cases of incomplete windows the windowed sentence will
be filled with Nones. The windowed tag sequence will be
filled with index 9 since the first 9 numbers (incl. 0)
are already reserved for the possible tags of the 
conll2003 dataset.
"""
def create_windows(sentences, tags, window_size):
    windowed_sents = []
    windowed_tags = []
    for sent, sent_tags in zip(sentences, tags):
        for i in range(0, len(sent), window_size):
            windowed_sent = sent[i:i + window_size]
            windowed_sent_tags = sent_tags[i:i + window_size]
            padding_size = window_size - len(windowed_sent)
            if padding_size > 0:
                windowed_sent += [None] * padding_size
                windowed_sent_tags += [9] * padding_size
            windowed_sents.append(windowed_sent)
            windowed_tags.append(windowed_sent_tags)
    return windowed_sents, windowed_tags

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

"""
Loads and unpickles the pickled word-to-index-dictionary.
"""
def load_word_index_dict_pickle():
    with open(WORD2INDEX_PATH, 'rb') as fd:
        word2index = pickle.load(fd)
    return word2index

"""
Creates a torch dataloader with a specific
batch size and a window size.
"""
def create_dataloader(path, word2index, batch_size=1, shuffle=False, window_size=100):
    sents, tags = get_data(path)
    w_sents, w_tags = create_windows(sents, tags, window_size)
    data = NERData(w_sents, w_tags, word2index)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader

# Run this script to create a word-to-index-dict-pickle if not existing so far!
if __name__ == '__main__':
    create_word_index_dict_pickle()
    # print(len(load_word_index_dict_pickle()))
