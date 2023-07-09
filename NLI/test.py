import torch
from torch.utils.data import DataLoader

from main import evaluate
from model import Clf_model, HParam, BEST_NLI_MODEL_PATH
from utils import load_word_index_dict_pickle, get_data, NLIData

"""
Tests the trained model on the test dataset.
"""
def test():
    word2index = load_word_index_dict_pickle()

    best_model = Clf_model(
        len(word2index),
        emb_dim=HParam.emb_dim,
        hidden_dim=HParam.hidden_dim,
        num_layers=HParam.num_layers,
        bidirectional=HParam.bidirectional
    )

    try:
        best_model.load_state_dict(torch.load(BEST_NLI_MODEL_PATH))

        test_sents_1, test_sents_2, test_labels = get_data("data/test.tsv")
        test_data = NLIData(test_sents_1, test_sents_2, test_labels, word2index)
        test_loader = DataLoader(test_data, batch_size=32)

        acc = evaluate(best_model, test_loader)
        print("test accuracy:", acc)
    except:
        print("No pretrained model existing!")

if __name__ == '__main__':
    test()
