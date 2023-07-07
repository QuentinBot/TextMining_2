import torch
import torch.nn as nn
from torch.utils.data import Dataset
from utils import get_data, word_2_index, get_vocab
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


class MyDataset(Dataset):
    def __init__(self, word2index, sents_1, sents_2, labels):
        self.max_length = 256
        self.labels = labels
        self.sentences = [sent_1 + " <sep> " + sent_2 for sent_1, sent_2 in zip(sents_1, sents_2)]
        for index, sent in enumerate(self.sentences):
            sent = [word2index[word] for word in sent.split()]
            if len(sent) < self.max_length:
                sent += [0] * (self.max_length - len(sent))
            else:
                sent = sent[:self.max_length]
            self.sentences[index] = sent

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        return torch.Tensor(sentence), torch.LongTensor([label])


train_path = "data/train.tsv"
val_path = "data/dev.tsv"
test_path = "data/test.tsv"

train_sents_1, train_sents_2, train_labels = get_data(train_path)
val_sents_1, val_sents_2, val_labels = get_data(val_path)
test_sents_1, test_sents_2, test_labels = get_data(test_path)

vocab = get_vocab(train_sents_1, train_sents_2, val_sents_1, val_sents_2, test_sents_1, test_sents_2)
word2index = word_2_index(vocab)

train_data = MyDataset(word2index, train_sents_1, train_sents_2, train_labels)
train_loader = DataLoader(train_data, batch_size=100)

val_data = MyDataset(word2index, val_sents_1, val_sents_2, val_labels)
val_loader = DataLoader(val_data, batch_size=100)


class RNNModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super(RNNModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available

embedding_dim = 256
hidden_dim = 128
num_classes = 2
model = RNNModel(embedding_dim, hidden_dim, num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 20  # Set the number of training epochs

for epoch in range(num_epochs):
    # Training
    model.train()
    for sentences, labels in train_loader:
        sentences = sentences.to(device)
        labels = labels.flatten(0, 1)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)

        # Forward pass
        outputs = model(sentences)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    preds = []
    original = []
    model.eval()
    with torch.no_grad():
        for sentences, labels in val_loader:
            sentences = sentences.to(device)
            labels = labels.to(device)

            outputs = model(sentences)
            _, predicted = torch.max(outputs, 1)
            # print(predicted)

            preds.extend(predicted.reshape(-1).tolist())
            original.extend(labels.reshape(-1).tolist())
        accuracy = accuracy_score(original, preds)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}")
