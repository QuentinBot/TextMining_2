# Architecture report

## Simple network with Elman RNN
model path: **models/rnn_model_nli.pt**

- unidirectional single layer Elman RNN
- embedding dimension: 100
- fully connected layer with hidden size of 256

Each word (index) of the sentence is mapped to an embedding vector. Together with a randomly initialized hidden state vector, the embedding vectors are passed to the Elman RNN. All these created hidden states are then passed to a fully connected layer. Lastly, the ReLU activation function is applied to the result.
