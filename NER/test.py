import torch

from main import evaluate
from model import RNN_model, HParam, BEST_MODEL_PATH
from utils import load_word_index_dict_pickle, create_dataloader

"""
Tests the model on the test dataset.
"""
def test():
    word2index = load_word_index_dict_pickle()
    test_loader = create_dataloader("data/conll2003_test.pkl", word2index, batch_size=100, window_size=100)
    
    best_model = RNN_model(len(word2index), emb_dim=HParam.embedding_dim, hidden_size=HParam.hidden_dim)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH))

    acc = evaluate(best_model, test_loader)
    print("test accuracy:", acc)

if __name__ == '__main__':
    test()
