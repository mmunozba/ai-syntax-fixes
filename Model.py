"""
Module containing Pytorch machine learning models.

"""

import torch
import torch.nn as nn

# Select CPU or GPU for Pytorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # Force CPU


class RNN(nn.Module):
    """
    Pytorch RNN model to be used for predictions on Python code token sequences.
    
    Parameters
    ----------
    input_size : int
        Size of the input vocabulary.
    hidden_size : int
        Size of the hidden field to be used by the GRU layer.
    output_size : int
        Size of the output vector.
    
    Attributes
    ----------
    hidden_size : int
        Size of the hidden field to be used by the GRU layer.
    embedding : Embedding
        Embedding layer for input embeddings.
    gru : GRU
        GRU NN layer.
    linear : Linear
        Linear NN layer.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=0.1)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        x, hidden = self.gru(embedded, hidden)
        output = self.linear(x)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
