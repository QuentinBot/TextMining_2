import torch
import torch.nn as nn

BEST_NLI_MODEL_PATH = "models/rnn_model_winograd.pt"

class Clf_model(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=1, bidirectional=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = 1 + int(bidirectional)

        # Embedding layer
        self.emb = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)

        # RNNs
        self.rnn = nn.RNN(self.emb_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layers, bidirectional=bidirectional)
        # self.gru = nn.GRU(self.emb_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layers, bidirectional=bidirectional)
        # self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layers, bidirectional=bidirectional)

        # Fully connected layers
        fc_dim = self.hidden_dim * self.bidirectional
        self.fc1 = nn.Linear(fc_dim, 128)
        self.fc2 = nn.Linear(128, 2)

        # Activation functions
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()
        # self.dropout = nn.Dropout(0.15)

    def forward(self, a, a_lengths, b, b_lengths):
        bs = a.size(0)

        a_emb = self.emb(a)
        b_emb = self.emb(b)
        
        a_packed = nn.utils.rnn.pack_padded_sequence(a_emb, a_lengths, batch_first=True, enforce_sorted=False)
        b_packed = nn.utils.rnn.pack_padded_sequence(b_emb, b_lengths, batch_first=True, enforce_sorted=False)
        # print(X.batch_sizes)
        h_0 = torch.rand(self.bidirectional * self.num_layers, bs, self.hidden_dim)
        packed_output, last_hidden = self.rnn(a_packed, h_0)
        packed_output, last_hidden = self.rnn(b_packed, last_hidden)
        # packed_output, (last_hidden, cell_state) = self.lstm(a_packed)
        # packed_output, (last_hidden, cell_state) = self.lstm(b_packed, (last_hidden, cell_state))
        # print(h_0.shape)
        # print(packed_output.data.shape)
        # print(last_hidden.shape)
        last_hidden = torch.permute(last_hidden, (1, 0, 2))
        last_hidden = last_hidden[:, -1 * self.bidirectional:, :].reshape(bs, -1)
        z_1 = self.relu(self.fc1(last_hidden))
        return self.fc2(z_1)

class HParam():
    emb_dim = 300
    hidden_dim = 512
    num_layers = 2
    bidirectional = True
