# Architecture report

## attempt 1: Elman RNN
- single layer Elman RNN
- hidden size: 256
- embedding dimension: 100

Each word (index) of the sentence is mapped to an embedding vector. Together with a randomly initialized hidden state vector, the embedding vectors are passed to the Elman RNN. All these created hidden states are then passed to a fully connected layer. Lastly, the ReLU activation function is applied to the result.
