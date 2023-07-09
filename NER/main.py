from tqdm import tqdm
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import RNN_model, HParam, BEST_MODEL_PATH
from utils import load_word_index_dict_pickle, create_dataloader, get_data, create_windows, NERData

"""
Trains the model on the given train dataset.
Also performs an evaluation of the model on
the given validation dataset after each 
training session per epoch. If the accuracy 
has increased, the model will be saved.
"""
def train(
        rnn_model: nn.Module,
        train_loader,
        val_loader,
        epochs=1,
        learning_rate=0.1
    ):
    optimizer = Adam(rnn_model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(
        # ignore_index=9
    )
    val_accuracy = 0

    for epoch in range(1, epochs + 1):
        print("epoch:", epoch)

        print("training model...")
        rnn_model.train()
        for batch in tqdm(train_loader):
            X: torch.Tensor = batch[0]
            y: torch.Tensor = batch[1]
            
            optimizer.zero_grad()

            out = rnn_model(X)
            loss: torch.Tensor = criterion(torch.log_softmax(out, -1).flatten(0, 1), y.flatten(0, 1))
            loss.backward()
            optimizer.step()
        
        acc = evaluate(rnn_model, val_loader)
        print("accuracy:", acc)
        if acc > val_accuracy:
            torch.save(rnn_model.state_dict(), BEST_MODEL_PATH)
            val_accuracy = acc
            print("model saved!")

"""
Evaluates the model on the given validation dataset.
"""
def evaluate(rnn_model: nn.Module, val_loader):
    preds = []
    labels = []

    print("evaluating model...")
    rnn_model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader):
            X: torch.Tensor = batch[0]
            y: torch.Tensor = batch[1]

            out = rnn_model(X)
            pred = torch.argmax(out, -1)
            preds.extend(pred.reshape(-1).tolist())
            labels.extend(y.reshape(-1).tolist())
    
    return accuracy_score(preds, labels)

def main():
    word2index = load_word_index_dict_pickle()
    train_loader = create_dataloader("data/conll2003_train.pkl", word2index, batch_size=100, shuffle=True, window_size=100)
    val_loader = create_dataloader("data/conll2003_val.pkl", word2index, batch_size=100, shuffle=True, window_size=100)
    
    rnn_model = RNN_model(len(word2index), emb_dim=HParam.embedding_dim, hidden_size=HParam.hidden_dim)
    train(rnn_model, train_loader, val_loader, epochs=10, learning_rate=0.0003)

"""
NOTE: For experimental purposes
"""
def test():
    word2index = load_word_index_dict_pickle()

    train_sents, train_tags = get_data("data/conll2003_train.pkl")
    w_train_sents, w_train_tags = create_windows(train_sents, train_tags, 100)
    train_data = NERData(w_train_sents, w_train_tags, word2index)
    train_loader = DataLoader(train_data, batch_size=100)
    
    val_sents, val_tags = get_data("data/conll2003_val.pkl")
    w_val_sents, w_val_tags = create_windows(val_sents, val_tags, 100)
    val_data = NERData(w_val_sents, w_val_tags, word2index)
    val_loader = DataLoader(val_data, batch_size=100)
    
    rnn_model = RNN_model(len(word2index), emb_dim=HParam.embedding_dim, hidden_size=HParam.hidden_dim)
    criterion = nn.NLLLoss(ignore_index=9)
    for batch in val_loader:
        X: torch.Tensor = batch[0]
        y: torch.Tensor = batch[1]
        # break
        out = rnn_model(X)
        pred = torch.log_softmax(out, -1).flatten(0, 1)
        labels = y.flatten(0, 1)
        print(out.shape)
        # print(labels.shape)
        loss = criterion(pred, labels)
        # print(loss)
        # pred = torch.argmax(out, -1)
        # print(pred.reshape(-1).tolist())
        # print(y.reshape(-1).tolist())
        break

if __name__ == '__main__':
    main()
