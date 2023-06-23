import torch
import torch.nn as nn

class RNN_model(nn.Module):
    def __init__(self, hidden_size, vocab_size, emb_dim):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        
        self.emb = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)

        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

        self.relu = nn.ReLU()
    
    def forward(self):
        pass
