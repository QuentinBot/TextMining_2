import torch
from torch.utils.data import DataLoader

from main import evaluate
from model import RNN_model, HParam, BEST_MODEL_PATH
from utils import load_word_index_dict_pickle, get_data, create_windows, NERData

"""
Tests the model on the test dataset.
"""
def test():
    word2index = load_word_index_dict_pickle()
    # print(len(word2index))

    test_sents, test_tags = get_data("data/conll2003_test.pkl")
    w_test_sents, w_test_tags = create_windows(test_sents, test_tags, 100)
    test_data = NERData(w_test_sents, w_test_tags, word2index)
    test_loader = DataLoader(test_data, batch_size=100)
    
    best_model = RNN_model(len(word2index), emb_dim=HParam.embedding_dim, hidden_size=HParam.hidden_dim)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH))

    acc = evaluate(best_model, test_loader)
    print("test accuracy:", acc)

if __name__ == '__main__':
    test()
