import torch
import torch.nn as nn

BEST_MODEL_PATH = "models/rnn_model_deep_conll.pt"

class RNN_model(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        
        # Embedding layer
        self.emb = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        # print(self.emb)

        # RNN Layer
        self.rnn = nn.RNN(self.emb_dim, self.hidden_size, batch_first=True)
        
        # Fully connected layers
        self.fc = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.fc2 = nn.Linear(self.hidden_size // 2, 10)

        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, xb: torch.Tensor):
        # print(torch.max(xb))
        embeddings = self.emb(xb)
        h_0 = torch.rand(1, xb.size(0), self.hidden_size)
        all_hidden_states, last_hidden_state = self.rnn(embeddings, h_0)
        z_0 = self.fc(all_hidden_states)
        out = self.relu(z_0)
        return self.fc2(out)

"""
Consists predefined hyperparameters
"""
class HParam():
    hidden_dim = 256
    embedding_dim = 100
