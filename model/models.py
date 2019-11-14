import random
import numpy as np
import torch
import torch.nn as nn
from base import BaseModel
from model.custom_layers.lstm_with_gaussian_attention import LSTMWithGaussianAttention
from model.custom_layers.seq2seq_modules import Encoder, Decoder


class UnconditionalHandwriting(BaseModel):
    """
    Class for Unconditional Handwriting generation
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_gaussian, dropout, char2idx, device):
        super(UnconditionalHandwriting, self).__init__()

        # Params
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gaussian = num_gaussian
        self.output_dim = 6 * num_gaussian + 1
        self.num_layers = num_layers
        self.char2idx = char2idx
        self.dropout = dropout

        # Setting an attribute device
        self.device = device

        # Define the RNN and the Mixture Density layers
        self.rnn_1 = nn.LSTM(input_size=input_dim,
                             hidden_size=hidden_dim,
                             num_layers=num_layers,
                             dropout=dropout,
                             batch_first=True)

        self.rnn_2 = nn.LSTM(input_size=input_dim + hidden_dim,
                             hidden_size=hidden_dim,
                             num_layers=num_layers,
                             dropout=dropout,
                             batch_first=True)

        self.rnn_3 = nn.LSTM(input_size=input_dim + hidden_dim,
                             hidden_size=hidden_dim,
                             num_layers=num_layers,
                             dropout=dropout,
                             batch_first=True)

        self.mixture_density_layer = nn.Linear(3 * self.hidden_dim, self.output_dim)

    def forward(self, sentences, sentences_mask, strokes, strokes_mask):

        # We don't need the sentences for the unconditional handwriting generation
        _, _, strokes, strokes_mask = sentences, sentences_mask, strokes, strokes_mask

        # Initialization of the hidden layers
        batch_size = strokes.size(0)
        hidden_1 = self.init_hidden(batch_size)
        hidden_2 = self.init_hidden(batch_size)
        hidden_3 = self.init_hidden(batch_size)

        # Fist rnn
        output_rnn_1, hidden_1 = self.rnn_1(strokes, hidden_1)
        # Second rnn
        input_rnn_2 = torch.cat([strokes, output_rnn_1], dim=-1)
        output_rnn_2, hidden_2 = self.rnn_2(input_rnn_2, hidden_2)
        # Third rnn
        input_rnn_3 = torch.cat([strokes, output_rnn_2], dim=-1)
        output_rnn_3, hidden_3 = self.rnn_3(input_rnn_3, hidden_3)
        # Application of the mixture density layer
        input_mdl  = torch.cat([output_rnn_1, output_rnn_2, output_rnn_3], dim=-1)
        output_mdl = self.mixture_density_layer(input_mdl)

        return output_mdl

    def init_hidden(self, batch_size=1):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        return hidden_state, cell_state

    def compute_gaussian_parameters(self, output_network, sampling_bias=0.):
        pi, mu1, mu2, sigma1, sigma2, rho, eos = output_network.split(self.num_gaussian, dim=2)

        # Normalization of the output + adding the bias
        pi = torch.softmax(pi * (1 + sampling_bias), dim=2)
        mu1 = mu1
        mu2 = mu2
        sigma1 = torch.exp(sigma1 - sampling_bias)
        sigma2 = torch.exp(sigma2 - sampling_bias)
        rho = torch.tanh(rho)
        eos = torch.sigmoid(eos)  # Equivalent to the normalization given in the paper

        return pi, mu1, mu2, sigma1, sigma2, rho, eos

    def generate_unconditional_sample(self, sampling_bias=1.):
        # Prepare input sequence
        stroke = [0., 0., 0.]  # init sample
        stroke = torch.tensor(stroke).view(1, 1, 3)  # (bs, seq_len, 3)
        list_strokes = []

        # Initialization of the hidden layers
        hidden_1 = self.init_hidden(stroke.size(0))
        hidden_2 = self.init_hidden(stroke.size(0))
        hidden_3 = self.init_hidden(stroke.size(0))

        with torch.no_grad():
            for i in range(700):  # sampling len

                # First rnn
                output_rnn_1, hidden_1 = self.rnn_1(stroke, hidden_1)
                # Second rnn
                input_rnn_2 = torch.cat([stroke, output_rnn_1], dim=-1)
                output_rnn_2, hidden_2 = self.rnn_2(input_rnn_2, hidden_2)
                # Third rnn
                input_rnn_3 = torch.cat([stroke, output_rnn_2], dim=-1)
                output_rnn_3, hidden_3 = self.rnn_3(input_rnn_3, hidden_3)
                # Application of the mixture density layer
                input_mdl = torch.cat([output_rnn_1, output_rnn_2, output_rnn_3], dim=-1)
                output_mdl = self.mixture_density_layer(input_mdl)
                # Computing the gaussian mixture parameters
                gaussian_params = self.compute_gaussian_parameters(output_mdl, sampling_bias)
                pi, mu1, mu2, sigma1, sigma2, rho, eos = gaussian_params

                # Sample the next stroke
                eos = torch.bernoulli(eos)         # Decide whether to stop or continue the stroke
                idx = torch.multinomial(pi[0], 1)  # Pick a gaussian with a multinomial law based on weights pi

                # Select the parameters of the picked gaussian
                mu1 = mu1[0, 0, idx]
                mu2 = mu2[0, 0, idx]
                sigma1 = sigma1[0, 0, idx]
                sigma2 = sigma2[0, 0, idx]
                rho = rho[0, 0, idx]

                # Sampling from a bivariate gaussian:
                z1 = torch.normal(mean=0., std=torch.ones(1)).view(1, 1, -1)
                z2 = torch.normal(mean=0., std=torch.ones(1)).view(1, 1, -1)
                x1 = sigma1 * z1 + mu1
                x2 = sigma2 * (rho * z1 + torch.sqrt(1 - rho ** 2) * z2) + mu2

                # Adding the stroke to the list and updating the stroke
                stroke = torch.cat([eos, x1, x2], 2)
                list_strokes.append(stroke.squeeze().numpy())
        return np.array(list_strokes)


class ConditionalHandwriting(BaseModel):
    """
    Class for Conditional Handwriting generation
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_gaussian_out, dropout, num_chars, num_gaussian_window,
                 char2idx, device):
        super(ConditionalHandwriting, self).__init__()

        # Params
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gaussian_out = num_gaussian_out
        self.num_gaussian_window = num_gaussian_window
        self.num_chars = num_chars
        self.output_dim = 6 * num_gaussian_out + 1
        self.num_layers = num_layers
        self.dropout = dropout
        self.char2idx = char2idx

        # Setting an attribute device
        self.device = device

        # Define RNN layers
        self.rnn_1_with_gaussian_attention = LSTMWithGaussianAttention(input_dim=input_dim,
                                                                       hidden_dim=hidden_dim,
                                                                       num_gaussian_window=num_gaussian_window,
                                                                       num_chars=num_chars,
                                                                       device=self.device)

        self.rnn_2 = nn.LSTM(input_size=input_dim + hidden_dim + num_chars,
                             hidden_size=hidden_dim,
                             num_layers=num_layers,
                             dropout=dropout,
                             batch_first=True)

        self.rnn_3 = nn.LSTM(input_size=input_dim + hidden_dim + num_chars,
                             hidden_size=hidden_dim,
                             num_layers=num_layers,
                             dropout=dropout,
                             batch_first=True)

        # Define the mixture density layer
        self.mixture_density_layer = nn.Linear(3 * self.hidden_dim, self.output_dim)

    def forward(self, sentences, sentences_mask, strokes, strokes_mask):

        # Initialization of the hidden layers for the rnn 2 & 3
        batch_size = strokes.size(0)
        hidden_2 = self.init_hidden(batch_size)
        hidden_3 = self.init_hidden(batch_size)

        # First rnn with gaussian attention
        output_rnn_1_attention, window, _ = self.rnn_1_with_gaussian_attention(strokes=strokes,
                                                                               sentences=sentences,
                                                                               sentences_mask=sentences_mask)
        # Second rnn
        input_rnn_2 = torch.cat([strokes, window, output_rnn_1_attention], dim=-1)
        output_rnn_2, hidden_2 = self.rnn_2(input_rnn_2, hidden_2)
        # Third rnn
        input_rnn_3 = torch.cat([strokes, window, output_rnn_2], dim=-1)
        output_rnn_3, hidden_3 = self.rnn_3(input_rnn_3, hidden_3)
        # Application of the mixture density layer
        input_mdl = torch.cat([output_rnn_1_attention, output_rnn_2, output_rnn_3], dim=-1)
        output_mdl = self.mixture_density_layer(input_mdl)

        return output_mdl

    def init_hidden(self, batch_size=1):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        return hidden_state, cell_state

    def compute_gaussian_parameters(self, output_network, sampling_bias=0.):
        pi, mu1, mu2, sigma1, sigma2, rho, eos = output_network.split(self.num_gaussian_out, dim=2)

        # Normalization of the output + adding the bias
        pi = torch.softmax(pi * (1 + sampling_bias), dim=2)
        mu1 = mu1
        mu2 = mu2
        sigma1 = torch.exp(sigma1 - sampling_bias)
        sigma2 = torch.exp(sigma2 - sampling_bias)
        rho = torch.tanh(rho)
        eos = torch.sigmoid(eos)  # Equivalent to the normalization given in the paper

        return pi, mu1, mu2, sigma1, sigma2, rho, eos

    def generate_conditional_sample(self, sentence, sampling_bias=1.):

        # Adding a space char to the sentence for computing the exit condition
        sentence += ' '

        # Transforming the sentence into a tensor
        sentence = torch.tensor(data=[self.char2idx[char] for char in sentence], dtype=torch.long)
        sentence_mask = torch.ones(sentence.shape)

        # Prepare input sequence
        stroke = [0., 0., 0.]  # init the sample
        stroke = torch.tensor(stroke).view(1, 1, 3)  # (bs, seq_len, 3)
        list_strokes = []

        # Set re_init = False in order to keep the hidden states during the loop
        self.rnn_1_with_gaussian_attention.re_init = False
        # Initialization of the hidden layer of the rnn 2 & 3
        hidden_2 = self.init_hidden(stroke.size(0))
        hidden_3 = self.init_hidden(stroke.size(0))

        with torch.no_grad():
            for i in range(700):  # sampling len

                # First rnn with gaussian attention
                output_rnn_1_attention, window, phi = self.rnn_1_with_gaussian_attention(strokes=stroke,
                                                                                         sentences=sentence,
                                                                                         sentences_mask=sentence_mask)
                # Second rnn
                input_rnn_2 = torch.cat([stroke, window, output_rnn_1_attention], dim=-1)
                output_rnn_2, hidden_2 = self.rnn_2(input_rnn_2, hidden_2)
                # Third rnn
                input_rnn_3 = torch.cat([stroke, window, output_rnn_2], dim=-1)
                output_rnn_3, hidden_3 = self.rnn_3(input_rnn_3, hidden_3)
                # Application of the mixture density layer
                input_mdl = torch.cat([output_rnn_1_attention, output_rnn_2, output_rnn_3], dim=-1)
                output_mdl = self.mixture_density_layer(input_mdl)
                # Computing the gaussian mixture parameters
                gaussian_params = self.compute_gaussian_parameters(output_mdl, sampling_bias)
                pi, mu1, mu2, sigma1, sigma2, rho, eos = gaussian_params

                # Exit condition
                if int(torch.argmax(phi)) + 1 == sentence.size(0):
                    break

                # Sample the next stroke
                eos = torch.bernoulli(eos)         # Decide whether to stop or continue the stroke
                idx = torch.multinomial(pi[0], 1)  # Pick a gaussian with a multinomial law based on weights pi

                # Select the parameters of the picked gaussian
                mu1 = mu1[0, 0, idx]
                mu2 = mu2[0, 0, idx]
                sigma1 = sigma1[0, 0, idx]
                sigma2 = sigma2[0, 0, idx]
                rho = rho[0, 0, idx]

                # Sampling from a bivariate gaussian:
                z1 = torch.normal(mean=0., std=torch.ones(1)).view(1, 1, -1)
                z2 = torch.normal(mean=0., std=torch.ones(1)).view(1, 1, -1)
                x1 = sigma1 * z1 + mu1
                x2 = sigma2 * (rho * z1 + torch.sqrt(1 - rho ** 2) * z2) + mu2

                # Adding the stroke to the list and updating the stroke
                stroke = torch.cat([eos, x1, x2], 2)
                list_strokes.append(stroke.squeeze().numpy())

        return np.array(list_strokes)


class Seq2SeqRecognition(BaseModel):
    """
    Class for Handwriting Recognition using a Seq2Seq with attention model
    """
    def __init__(self, encoder_input_dim, hidden_dim, num_layers, dropout, num_chars, embed_char_dim,
                 teacher_forcing_ratio, char2idx, device):
        super(Seq2SeqRecognition, self).__init__()

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.num_layers = num_layers
        self.device = device

        # Adding sos token to vocab
        char2idx['<sos>'] = len(char2idx) + 1
        self.char2idx = char2idx
        self.num_chars = num_chars + 1

        self.encoder = Encoder(input_dim=encoder_input_dim,
                               hidden_dim=hidden_dim,
                               num_layers=num_layers,
                               dropout=dropout,
                               device=device)
        self.decoder = Decoder(embed_dim=embed_char_dim,
                               hidden_dim=hidden_dim,
                               num_chars=self.num_chars,
                               num_layers=num_layers,
                               dropout=dropout,
                               device=device)

    def forward(self, sentences, sentences_mask, strokes, strokes_mask):

        # Add sos tokens to the sentences
        batch_size = sentences.size(0)
        sos_tensor = torch.tensor([[self.char2idx['<sos>']] for i in range(batch_size)], device=self.device)
        sentences = torch.cat([sos_tensor, sentences], dim=-1)
        max_len = sentences.size(1)

        # init outputs tensor
        outputs = torch.zeros(batch_size, max_len, self.num_chars, device=self.device)

        encoder_output, hidden = self.encoder(strokes)
        hidden = hidden[:self.num_layers]
        output = sentences[:, 0]  # sos tokens
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)
            outputs[:, t, :] = output  # (bs, T, num chars)
            is_teacher = random.random() < self.teacher_forcing_ratio
            predicted_char = output.max(dim=1)[1]
            output = sentences[:, t] if is_teacher else predicted_char

        return outputs  # (bs, char_seq_len, num_chars)

    def recognize_sample(self, stroke, max_len=20):
        stroke = torch.tensor(stroke, device=self.device)
        strokes = stroke.unsqueeze(0)  # (bs=1, stroke_seq_len, 3)
        outputs = []  # predicted characters

        with torch.no_grad():
            encoder_output, hidden = self.encoder(strokes)
            hidden = hidden[:self.num_layers]
            output = torch.tensor([self.char2idx['<sos>'] for i in range(1)], device=self.device)  # sos tokens
            for t in range(1, max_len):
                output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)
                predicted_char = output.max(dim=1)[1]
                output = predicted_char
                # Exit condition
                if int(predicted_char[0]) == 0:  # <eos> token
                    break
                outputs.append(int(predicted_char[0]))

        return outputs

