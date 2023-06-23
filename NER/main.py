import torch
import torch.nn as nn

from utils import *

from tqdm import tqdm
from torch.optim import Adam

def train(
        rnn_model: nn.Module,
        train_loader,
        val_loader,
        epochs=10,
        learning_rate=0.1,
        batch_size=16
    ):
    optimizer = Adam(rnn_model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    val_accuracy = 0

    for epoch in range(epochs):
        rnn_model.train()
        for X, y in tqdm(train_loader):
            optimizer.zero_grad()

            output = rnn_model(X)
            loss: torch.Tensor = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        rnn_model.eval()
        acc = evaluate(rnn_model, val_loader)
        if acc > val_accuracy:
            torch.save(rnn_model.state_dict(), 'rnn_model_conll.pt')

"""
TODO: finish it!
"""
def evaluate(rnn_model, val_loader):
    with torch.no_grad():
        for X, y in val_loader:
            output = rnn_model(X)
    
    return 0

def main():
    sents, tags = get_data("data/conll2003_val.pkl")
    print(sents[0], tags[0])

if __name__ == '__main__':
    main()
