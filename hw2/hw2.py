import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os  # IMPORT OS for checking file existence
import collections  # IMPORT collections for building vocabulary
import time  # IMPORT time for tracking training duration
##### DEVICE - NVIDIA RTX 1650 Q EDITION #####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --- Helper Functions ---
def _read_words(filename):
    """Reads a file and returns a list of words, replacing newlines with <eos>."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Could not find file: {filename}")
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split() # give a list of words


def build_vocab(filename):
    """Builds a dictionary mapping words to integer IDs based on the training data."""
    data = _read_words(filename)
    counter = collections.Counter(data)
    # Sort by frequency (most frequent words first)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))# list of tuples (word, count) sorted by frequency

    words, _ = list(zip(*count_pairs))# 2 tuples: (words,), (counts,)
    word_to_id = dict(zip(words, range(len(words))))# mapping word to unique integer ID
    return word_to_id


def file_to_word_ids(filename, word_to_id):
    """Converts a file of text into a list of integers."""
    data = _read_words(filename)# list of words
    return [word_to_id[word] for word in data if word in word_to_id] # list of integer IDs if not in vocab, skip


# --- Loading data ---
def load_all_data(data_path=None):
    """
    Loads Train, Validation, and Test data from the specified path.
    """
    # 1. Define file paths
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    print("Building vocabulary from training data...")
    word_to_id = build_vocab(train_path)
    vocab_size = len(word_to_id)
    print(f"Vocabulary built! Size: {vocab_size} words.")

    # 2. Convert all datasets to list of integers
    print("Converting text to integers...")
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)

    return train_data, valid_data, test_data, vocab_size, word_to_id

data_path = r"C:\Users\amira\OneDrive\Desktop\PycharmProjects\main-python-project\Deep-Learning-course\hw2\data_ptb"

# bring in all the data
train_data, valid_data, test_data, vocab_size, word_to_id = load_all_data(data_path)

# Hyperparameters
num_layers = 2
dropout = 0.5
unroll_steps = 35
hidden_states = 200 # parameters are initialized uniformly in [−0.05, 0.05]
minibatch_size = 20
vocab_size = vocab_size
def batch_normalize(x, gamma, beta, eps=1e-5):

#### MODEL DEFINITION ####
class PTBModel(nn.Module):
    def __init__(self, vocab_size, hidden_states=200, num_layers=2,use_dropout: bool = False,,dropout=0.5, rnn_type='LSTM'):
        super(PTBModel,self).__init__()
        self.use_dropout = use_dropout
        self.encoder = nn.Embedding(vocab_size, hidden_states)
        self.rnn_type = rnn_type
        self.hidden_states = hidden_states
        self.num_layers = num_layers

        # Switch between LSTM and GRU as required by the assignment
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_states, hidden_states, num_layers, dropout=dropout, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(hidden_states, hidden_states, num_layers, dropout=dropout, batch_first=True)

        self.decoder = nn.Linear(hidden_states, vocab_size)

        self.init_weights()

        # Dropout
        if use_dropout:
            self.dropout = nn.Dropout(dropout)

    def init_weights(self):
        initrange = 0.05

        # 1. Initialize Encoder (Embedding)
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # 2. Initialize Decoder (Linear)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

        # 3. Initialize LSTM/GRU Weights (Crucial for Zaremba implementation)
        # PyTorch defaults usually don't match the Zaremba paper, so we force it here.
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                param.data.uniform_(-initrange, initrange)
            elif 'bias' in name:
                param.data.zero_()  # Biases are typically initialized to 0

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)

        # Reshape for decoding: (batch * sequence_length, hidden_states)
        decoded = self.decoder(output.reshape(output.size(0) * output.size(1), output.size(2)))

        # Return reshaped output: (batch, sequence_length, vocab_size)
        return decoded.reshape(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.num_layers, batch_size, self.hidden_states),
                    weight.new_zeros(self.num_layers, batch_size, self.hidden_states))
        else:
            return weight.new_zeros(self.num_layers, batch_size, self.hidden_states)


# Helper to detach hidden states from the history graph
def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(model, data_source, criterion, batch_size=20, num_steps=20):
    model.eval()
    total_loss = 0.
    total_len = 0
    hidden = model.init_hidden(batch_size)

    with torch.no_grad():
        # ptb_iterator is the function I provided in the previous turn
        for x, y in ptb_iterator(data_source, batch_size, num_steps):
            data = torch.from_numpy(x).long().to(device)
            targets = torch.from_numpy(y).long().to(device)

            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)

            # Flatten output and targets for CrossEntropyLoss
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item() * data.size(0) * data.size(1)  # Multiply by batch*seq_len
            total_len += data.size(0) * data.size(1)

    return np.exp(total_loss / total_len)  # Perplexity = exp(cross_entropy)


def train_model(model, train_data, valid_data, epochs=13, lr=1.0, batch_size=20, num_steps=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print(f"Starting training: {model.rnn_type} | Dropout: {model.drop.p}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.
        start_time = time.time()
        hidden = model.init_hidden(batch_size)

        # Decay learning rate if needed (simple schedule)
        if epoch > 4:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (0.5 ** (epoch - 4))

        for i, (x, y) in enumerate(ptb_iterator(train_data, batch_size, num_steps)):
            data = torch.from_numpy(x).long().to(device)
            targets = torch.from_numpy(y).long().to(device)

            # 1. Starting each batch, we detach the hidden state from how it was produced.
            #    We only need its values, not its gradient history.
            hidden = repackage_hidden(hidden)

            model.zero_grad()
            output, hidden = model(data, hidden)

            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            loss.backward()

            # Clip gradients to prevent explosion (Zaremba et al. uses 5 or 0.25)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            total_loss += loss.item()

        # End of epoch stats
        val_ppl = evaluate(model, valid_data, criterion, batch_size, num_steps)
        print(f'| Epoch {epoch + 1:3d} | Valid PPL {val_ppl:8.2f} | Time {time.time() - start_time:.1f}s')

    return model

# Configuration 1: LSTM without Dropout [cite: 80]
model_lstm_no_drop = PTBModel(vocab_size, hidden_size=200, dropout=0.0, rnn_type='LSTM').to(device)
train_model(model_lstm_no_drop, train_data, valid_data, epochs=13, lr=1.0)

# Configuration 2: LSTM with Dropout [cite: 81]
# Note: You may need more epochs or lower LR for dropout models
model_lstm_drop = PTBModel(vocab_size, hidden_size=200, dropout=0.5, rnn_type='LSTM').to(device)
train_model(model_lstm_drop, train_data, valid_data, epochs=20, lr=1.0)

# Configuration 3: GRU without Dropout [cite: 82]
model_gru_no_drop = PTBModel(vocab_size, hidden_size=200, dropout=0.0, rnn_type='GRU').to(device)
train_model(model_gru_no_drop, train_data, valid_data, epochs=13, lr=1.0)

# Configuration 4: GRU with Dropout [cite: 83]
model_gru_drop = PTBModel(vocab_size, hidden_size=200, dropout=0.5, rnn_type='GRU').to(device)
train_model(model_gru_drop, train_data, valid_data, epochs=20, lr=1.0)