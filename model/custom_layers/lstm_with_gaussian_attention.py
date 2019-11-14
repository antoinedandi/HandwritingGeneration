import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMWithGaussianAttention(nn.Module):
    """
    Implements a custom LSTM that uses its output at time t to compute the gaussian attention window at time step t
    and then pass the computed window as an argument in the LSTM at time step t+1
    Cf equation (52) in the paper by A. Graves
    """

    def __init__(self, input_dim, hidden_dim, num_gaussian_window, num_chars, device):
        super(LSTMWithGaussianAttention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gaussian_window = num_gaussian_window
        self.num_chars = num_chars
        self.device = device

        self.lstm_cell = nn.LSTMCell(input_size=input_dim + self.num_chars,
                                     hidden_size=hidden_dim)

        self.window_layer = nn.Linear(self.hidden_dim, 3 * self.num_gaussian_window)
        self.hidden, self.window_params, self.window = self.init_hidden_and_window()
        self.re_init = True  # Re initialize the hidden and window params when forward is called

    def forward(self, strokes, sentences, sentences_mask):
        batch_size, strokes_seq_len, _ = strokes.size()

        # The list containing the hidden states and the window states
        hidden_seq = []
        window_seq = []
        phi_seq = []

        # If training : initialization of the hidden state, of the window params and of the window
        if self.re_init:
            self.hidden, self.window_params, self.window = self.init_hidden_and_window(batch_size)

        for t in range(strokes_seq_len):

            # Prepare the input for the lstm cell
            x_t = strokes[:, t, :]
            input_t = torch.cat([x_t, self.window], dim=-1)

            # Compute the new hidden state of the lstm cell
            self.hidden = self.lstm_cell(input_t, self.hidden)  # 2 * (bs, hidden_dim)

            # Compute the new gaussian attention window
            self.window_params = self.compute_window_parameters(self.hidden, self.window_params)
            self.window, phi = self.compute_window(sentences, sentences_mask, self.window_params)   # (bs, num_chars)

            hidden_seq.append(self.hidden[0])
            window_seq.append(self.window)
            phi_seq.append(phi)

        hidden_seq_tensor = torch.stack(hidden_seq, dim=1)   # (bs, strokes_seq_len, hidden_dim)
        window_seq_tensor = torch.stack(window_seq, dim=1)   # (bs, strokes_seq_len, num_chars)
        phi_seq_tensor = torch.stack(phi_seq, dim=1)         # (bs, strokes_seq_len, chars_seq_len)

        return hidden_seq_tensor, window_seq_tensor, phi_seq_tensor

    def init_hidden_and_window(self, batch_size=1):

        # init hidden
        hidden_t = (torch.zeros(batch_size, self.hidden_dim, device=self.device),
                    torch.zeros(batch_size, self.hidden_dim, device=self.device))

        # init window params
        alpha_t = torch.zeros(batch_size, self.num_gaussian_window, device=self.device)
        beta_t  = torch.zeros(batch_size, self.num_gaussian_window, device=self.device)
        kappa_t = torch.zeros(batch_size, self.num_gaussian_window, device=self.device)
        window_params_t = (alpha_t, beta_t, kappa_t)

        # init window
        window_t = torch.zeros(batch_size, self.num_chars, device=self.device)

        return hidden_t, window_params_t, window_t

    def compute_window_parameters(self, hidden_t, window_params_t):
        window_params_hat = self.window_layer(hidden_t[0])
        alpha_hat, beta_hat, kappa_hat = window_params_hat.split(self.num_gaussian_window, dim=1)
        previous_kappa = window_params_t[-1]

        # Normalization of the params according to equations (49), (50), (51)
        alpha = torch.exp(alpha_hat)  # (bs, K)
        beta = torch.exp(beta_hat)
        kappa = previous_kappa + torch.exp(kappa_hat)

        return alpha, beta, kappa

    def compute_window(self, sentences, sentences_mask, window_params):
        alpha, beta, kappa = window_params
        char_seq_len = sentences.size(-1)

        # Prepare tensors for broadcasting, target shape = (bs, K, chars_seq_len)
        alpha = alpha.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        kappa = kappa.unsqueeze(-1)
        u_matrix = torch.arange(0, char_seq_len, dtype=torch.float, device=self.device).reshape(1, 1, -1)
        sentences_mask = sentences_mask.unsqueeze(-1)

        # Compute phi (bs, chars_seq_len)
        phi = alpha * torch.exp(-beta * (kappa - u_matrix) ** 2)
        phi = phi.sum(1).unsqueeze(-1)

        # Compute the one hot encoding of the character sequence (bs, chars_seq_len, num_chars)
        char_seq_encoding = F.one_hot(sentences, num_classes=self.num_chars).float()

        window = phi * char_seq_encoding  # (bs, char_seq_len, num_chars)
        window = window * sentences_mask.float()  # Apply the sentences mask
        window = window.sum(1)  # (bs, num_chars)

        return window, phi.squeeze(2)
