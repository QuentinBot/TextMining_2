from tqdm import tqdm
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import RNN_model, BEST_NLI_PATH
from utils import get_data, NLIData, word_2_index, get_vocab


def train(rnn_model: nn.Module, train_loader, val_loader, epochs=1, learning_rate=0.001):
    optimizer = Adam(rnn_model.parameters(), lr=learning_rate)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.NLLLoss()

    best_val_acc = 0

    for e in range(epochs):
        print("new epoch")
        rnn_model.train()

        for batch in tqdm(train_loader):
            sent, labels = batch
            optimizer.zero_grad()
            outputs = rnn_model(sent)
            output_labels = torch.log_softmax(outputs, dim=2)
            loss = criterion(output_labels.squeeze(0), labels)

            loss.backward()
            optimizer.step()

        acc = evaluate(rnn_model, val_loader)

        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(rnn_model, BEST_NLI_PATH)
            print(f"Accuracy: {acc} -> model saved")
        else:
            print(f"Accuracy: {acc}")


def evaluate(rnn_model, val_loader):
    preds = []
    original = []

    rnn_model.eval()

    with torch.no_grad():
        for batch in val_loader:
            sent, labels = batch

            outputs = rnn_model(sent)
            pred_label = torch.argmax(outputs, -1)
            print(pred_label)
            preds.extend(pred_label.reshape(-1).tolist())
            original.extend(labels.reshape(-1).tolist())

    # print(pred_label)
    # print(original)
    return accuracy_score(original, preds)


def main():
    train_path = "data/train.tsv"
    val_path = "data/dev.tsv"
    test_path = "data/test.tsv"

    EMB_DIM = 100
    HID_SIZE = 512

    train_sents_1, train_sents_2, train_labels = get_data(train_path)
    val_sents_1, val_sents_2, val_labels = get_data(val_path)
    test_sents_1, test_sents_2, test_labels = get_data(test_path)

    vocab = get_vocab(train_sents_1, train_sents_2, val_sents_1, val_sents_2, test_sents_1, test_sents_2)
    word2index = word_2_index(vocab)

    train_data = NLIData(word2index, train_sents_1, train_sents_2, train_labels)
    train_loader = DataLoader(train_data, batch_size=100)

    val_data = NLIData(word2index, val_sents_1, val_sents_2, val_labels)
    val_loader = DataLoader(val_data, batch_size=100)

    rnn_model = RNN_model(len(word2index), EMB_DIM, HID_SIZE)
    train(rnn_model, train_loader, val_loader, epochs=15, learning_rate=0.01)


if __name__ == "__main__":
    main()
