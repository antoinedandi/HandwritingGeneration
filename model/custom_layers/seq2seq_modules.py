import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder class for the Seq2Seq handwriting recognition model
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, device):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.rnn = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          dropout=dropout,
                          bidirectional=True,
                          batch_first=True)

    def forward(self, input_batch, hidden=None):
        output_rnn, hidden = self.rnn(input_batch, hidden)
        # summing the bidirectional outputs
        output_rnn = (output_rnn[:, :, :self.hidden_dim] + output_rnn[:, :, self.hidden_dim:])
        return output_rnn, hidden   # (bs, T, hidden_dim), (2, bs, hidden_dim)

    def init_hidden(self, batch_size=1):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        return hidden_state


class Attention(nn.Module):
    """
    Attention class used by the Decoder for the Seq2Seq handwriting recognition model
    """
    def __init__(self, hidden_dim, device):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.attn = nn.Linear(self.hidden_dim * 2, hidden_dim)
        self.V = nn.Parameter(torch.randn(hidden_dim) / math.sqrt(hidden_dim))

    def forward(self, hidden, encoder_outputs):

        # Reshaping tensors for bmm
        batch_size, stroke_seq_len = encoder_outputs.size(0), encoder_outputs.size(1)
        hidden = hidden.repeat(stroke_seq_len, 1, 1).transpose(0, 1)  # (bs, T, hidden size)

        # Calculating alignment scores
        x = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # (bs, T, hidden)
        V = self.V.repeat(batch_size, 1).unsqueeze(2)          # (bs, hidden, 1)
        alignment_scores = x.bmm(V)  # (bs, T, 1)

        # Applying softmax on alignment scores to get Attention weights
        attn_weights = torch.softmax(alignment_scores, dim=1)  # (bs, T, 1)

        return attn_weights.transpose(1, 2)  # (bs, 1, T)


class Decoder(nn.Module):
    """
    Decoder class for the Seq2Seq handwriting recognition model
    """
    def __init__(self, embed_dim, hidden_dim, num_chars, num_layers, dropout, device):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_chars = num_chars
        self.num_layers = num_layers
        self.device = device

        self.embed = nn.Embedding(num_chars, embed_dim)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_dim, device)
        self.gru = nn.GRU(input_size=hidden_dim + embed_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          dropout=dropout,
                          batch_first=True)
        self.out = nn.Linear(hidden_dim * 2, num_chars)

    def forward(self, last_chars, last_hidden, encoder_outputs):

        # Get the embedding of the current input char
        embedded_char = self.embed(last_chars).unsqueeze(1)  # (bs,1,N)
        embedded_char = self.dropout(embedded_char)

        # Compute the context
        attention_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = torch.bmm(attention_weights, encoder_outputs)  # (bs,1,hidden)

        # Combine embedded input char and context and pass it in the rnn
        rnn_input = torch.cat([embedded_char, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(1)  # (bs,1,hidden)
        context = context.squeeze(1)  # (bs,hidden)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)

        return output, hidden, attention_weights
