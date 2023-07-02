import torch
from torch.utils.data import Dataset
import pandas as pd
from nltk.corpus import stopwords
import nltk
import string


class NLIData(Dataset):

    def __init__(self, word2index, sents_1, sents_2, labels):
        self.sents_1 = sents_1
        self.sents_2 = sents_2
        self.labels = labels
        self.max_char = 256
        self.word2index = word2index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sent_1 = self.sents_1[index].split(" ")
        sent_1 = [self.word2index[word] for word in sent_1]
        sent_2 = self.sents_2[index].split(" ")
        sent_2 = [self.word2index[word] for word in sent_2]
        label = self.labels[index]

        if len(sent_1) > self.max_char:
            sent_1 = sent_1[:self.max_char]
        else:
            sent_1.extend([0 for _ in range(self.max_char - len(sent_1))])

        if len(sent_2) > self.max_char:
            sent_2 = sent_2[:self.max_char]
        else:
            sent_2.extend([0 for _ in range(self.max_char - len(sent_2))])

        return torch.LongTensor(sent_1), torch.LongTensor(sent_2), torch.LongTensor([label])


def word_2_index(vocab):
    word2index = {}
    for index, word in enumerate(vocab):
        word2index[word] = index + 1

    return word2index


def get_data(path):
    # nltk.download("stopwords")

    data = pd.read_csv(path, sep="\t")
    sents_1 = data["sentence1"].values.tolist()
    sents_1 = [preprocess_sentence(sent) for sent in sents_1]
    sents_2 = data["sentence2"].values.tolist()
    sents_2 = [preprocess_sentence(sent) for sent in sents_2]
    labels = data["label"].values.tolist()
    return sents_1, sents_2, labels


def preprocess_sentence(sent):

    # first try without removing stopwords because personal pronouns will be important for the task
    # stop_words = set(stopwords.words("english"))
    tokens = nltk.word_tokenize(sent)
    # filtered_sentence = [word.lower() for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
    filtered_sentence = [word.lower() for word in tokens if word.lower() not in string.punctuation]
    processed_sentence = " ".join(filtered_sentence)
    return processed_sentence


def get_vocab(t_1, t_2, v_1, v_2, e_1, e_2):
    vocab = []
    all_sents_list = t_1 + t_2 + v_1 + v_2 + e_1 + e_2

    for sent in all_sents_list:
        for word in sent.split(" "):
            if word not in vocab:
                vocab.append(word)

    return vocab
