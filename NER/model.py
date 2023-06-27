import torch
import torch.nn as nn

BEST_MODEL_PATH = "data/rnn_model_conll.pt"

class RNN_model(nn.Module):
    def __init__(self, hidden_size, vocab_size, emb_dim):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        
        # Embedding layer
        self.emb = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        # print(self.emb)

        # RNN Layer
        self.rnn = nn.RNN(self.emb_dim, self.hidden_size)
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_size, 10)

        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, xb):
        embeddings = self.emb(xb)
        h_0 = torch.rand(1, self.emb_dim, self.hidden_size)
        # print(xb.shape)
        # print(embeddings.shape)
        # print(h_0.shape)
        all_hidden_states, last_hidden_state = self.rnn(embeddings, h_0)
        z_0 = self.fc(all_hidden_states)
        out = self.relu(z_0)
        return out
