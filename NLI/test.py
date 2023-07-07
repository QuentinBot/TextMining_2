from utils import get_data, get_vocab, word_2_index, NLIData
from model import RNN_model, HParam, BEST_NLI_PATH
from main import evaluate

from torch.utils.data import DataLoader
import torch


def test():
    train_path = "data/train.tsv"
    val_path = "data/dev.tsv"
    test_path = "data/test.tsv"

    train_sents_1, train_sents_2, train_labels = get_data(train_path)
    val_sents_1, val_sents_2, val_labels = get_data(val_path)
    test_sents_1, test_sents_2, test_labels = get_data(test_path)

    vocab = get_vocab(train_sents_1, train_sents_2, val_sents_1, val_sents_2, test_sents_1, test_sents_2)
    word2index = word_2_index(vocab)

    test_data = NLIData(word2index, test_sents_1, test_sents_2, test_labels)
    test_loader = DataLoader(test_data, batch_size=100)

    best_model = RNN_model(len(word2index), emb_dim=HParam.embedding_dim, hidden_size=HParam.hidden_size)
    best_model.load_state_dict(torch.load(BEST_NLI_PATH))

    acc = evaluate(best_model, test_loader)
    print(f"Test accuracy: {acc}")


if __name__ == '__main__':
    test()
