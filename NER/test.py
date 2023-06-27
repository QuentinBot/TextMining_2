import torch
from torch.utils.data import DataLoader

from main import evaluate
from model import RNN_model, BEST_MODEL_PATH
from utils import load_word_index_dict_pickle, get_data, create_windows, NERData

"""
Tests the model on the test dataset.
"""
def test():
    word2index = load_word_index_dict_pickle()

    test_sents, test_tags = get_data("data/conll2003_test.pkl")
    w_test_sents, w_test_tags = create_windows(test_sents, test_tags, 100)
    test_data = NERData(w_test_sents, w_test_tags, word2index)
    test_loader = DataLoader(test_data, batch_size=100)
    
    best_model = RNN_model()
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH))

    acc = evaluate(best_model, test_loader)
    print("test accuracy:", acc)

if __name__ == '__main__':
    test()
