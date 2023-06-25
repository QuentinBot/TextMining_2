import torch
import torch.nn as nn

from utils import *

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader

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
    word2index = load_word_index_dict_pickle()
    sents, tags = get_data("data/conll2003_val.pkl")
    windowed_sents, windowed_tags = create_windows(sents, tags, 100)
    # print(len(windowed_sents[0]))
    # print(len(windowed_tags[0]))
    # return
    val_data = NERData(windowed_sents, windowed_tags, word2index)
    # X, y = val_data[0]
    # print(X.shape, y.shape)
    # print(X)
    # print(y)
    # return
    val_loader = DataLoader(val_data, batch_size=16)
    for X, y in val_loader:
        print(X.shape, y.shape)
        break

if __name__ == '__main__':
    main()
