import torch
import torch.nn as nn

BEST_NLI_PATH = "models/rnn_model_nli.pt"


class RNN_model(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        self.emb = nn.Embedding(self.vocab_size+1, self.emb_dim, padding_idx=0)

        self.rnn = nn.RNN(self.emb_dim, self.hidden_size, batch_first=True)

        self.lin1 = nn.Linear(self.hidden_size, 2)

        self.relu = nn.ReLU()

    def forward(self, inp):
        inp = self.emb(inp)
        h_0 = torch.rand(1, inp.size(0), self.hidden_size)
        all_hidden_states, last_hidden_state = self.rnn(inp, h_0)
        out = self.lin1(last_hidden_state)
        out = self.relu(out)
        return out


class HParam():
    hidden_size = 256
    embedding_dim = 100

