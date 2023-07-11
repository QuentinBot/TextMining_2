# architecture report

## attempt 42 *(after many attempts not listed here!)*: Deep network with Elman RNN
model path: **models/rnn_model_winograd.pt**

- bidirectional double layer Elman RNN
- embedding dimension: 300
- first fully connected layer with hidden size of (512 x 2)
- second fully connected layer with hidden size of 128

Each word (index) of the sentences is mapped to an embedding vector. Together with a randomly initialized hidden state vector, the embedding vectors are passed to the Elman RNN. Only the last hidden states are then passed to the first fully connected layer, followed by a ReLU activation. Lastly, the result is passed to the second fully connected layer.
