import torch

from main import evaluate
from model import RNN_model

def test():
    best_model = RNN_model()
    best_model.load_state_dict(torch.load('rnn_model_conll.pt'))
