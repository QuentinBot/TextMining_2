# Architecture report

## attempt 1: Simple network with Elman RNN *(deprecated)*
model path: **models/rnn_model_conll.pt**

- unidirectional single layer Elman RNN
- embedding dimension: 100
- fully connected layer with hidden size of 256

Each word (index) of the sentence is mapped to an embedding vector. Together with a randomly initialized hidden state vector, the embedding vectors are passed to the Elman RNN. All these created hidden states are then passed to a fully connected layer. Lastly, the ReLU activation function is applied to the result.

## attempt 2: Deep network with Elman RNN
model path: **models/rnn_model_deep_conll.pt**

- unidirectional single layer Elman RNN
- embedding dimension: 100
- first fully connected layer with hidden size of 256
- second fully connected layer with half hidden size (128)

Each word (index) of the sentence is mapped to an embedding vector. Together with a randomly initialized hidden state vector, the embedding vectors are passed to the Elman RNN. All these created hidden states are then passed to the first fully connected layer, followed by an activation process with the ReLU function. Lastly the result is passed to a second fully connected layer.
