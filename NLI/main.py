from tqdm import tqdm
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from model import Clf_model, HParam
from utils import load_word_index_dict_pickle, get_data, NLIData

def main():
    word2index = load_word_index_dict_pickle()
    # print(len(word2index))

    train_sents_1, train_sents_2, train_labels = get_data("data/train.tsv")
    train_data = NLIData(train_sents_1, train_sents_2, train_labels, word2index)
    train_loader = DataLoader(train_data, batch_size=32)

    val_sents_1, val_sents_2, val_labels = get_data("data/dev.tsv")
    val_data = NLIData(val_sents_1, val_sents_2, val_labels, word2index)
    val_loader = DataLoader(val_data, batch_size=32)

    rnn_model = Clf_model(
        len(word2index), 
        emb_dim=HParam.emb_dim, 
        hidden_dim=HParam.hidden_dim, 
        num_layers=HParam.num_layers, 
        bidirectional=HParam.bidirectional
    )
    train(rnn_model, train_loader, val_loader, epochs=20, learning_rate=0.001)
    # for batch in train_loader:
    #     a, a_len, b, b_len, labels = batch
    #     # print(a)
    #     print(a_len)
    #     out = rnn_model(a, a_len, b, b_len)
    #     # print(torch.argmax(out, -1))
    #     break

def train(
        rnn_model: nn.Module,
        train_loader,
        val_loader,
        epochs=1,
        learning_rate=0.1
    ):
    optimizer = Adam(rnn_model.parameters(), lr=learning_rate)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.NLLLoss(

    )
    val_accuracy = 0

    for epoch in range(1, epochs + 1):
        print("epoch:", epoch)

        print("training model...")
        rnn_model.train()
        for batch in tqdm(train_loader):
            a: torch.Tensor = batch[0]
            a_len: torch.Tensor = batch[1]
            b: torch.Tensor = batch[2]
            b_len: torch.Tensor = batch[3]
            labels: torch.Tensor = batch[4]

            optimizer.zero_grad()

            out = rnn_model(a, a_len, b, b_len)
            # print(out.shape)
            # print(torch.log_softmax(out, -1).squeeze(1).shape)
            # return
            loss: torch.Tensor = criterion(torch.log_softmax(out, -1), labels)
            # loss = criterion(out.squeeze(1), labels.float())
            loss.backward()
            optimizer.step()
        
        acc = evaluate(rnn_model, val_loader)
        print("accuracy:", acc)

def evaluate(rnn_model: nn.Module, val_loader):
    preds = []
    targets = []

    print("evaluating model...")
    rnn_model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader):
            a: torch.Tensor = batch[0]
            a_len: torch.Tensor = batch[1]
            b: torch.Tensor = batch[2]
            b_len: torch.Tensor = batch[3]
            labels: torch.Tensor = batch[4]

            out = rnn_model(a, a_len, b, b_len)
            pred = torch.argmax(out, -1)
            # pred = torch.round(out).int().reshape(-1)
            # print(labels.tolist())
            # print(pred.tolist())
            preds.extend(pred.reshape(-1).tolist())
            targets.extend(labels.reshape(-1).tolist())
    
    return accuracy_score(preds, targets)

if __name__ == '__main__':
    main()
