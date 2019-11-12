import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model.custom_layers.lstm_with_gaussian_attention import LSTMWithGaussianAttention


# Changes in case their is a problem:
# use self.hidden pour pas avoir à passer le hidden dans l'input
# change of the eos loss (sum over the good dimension)
# add norm layer

# TODO : mef dropout: x = F.dropout(x, training=self.training)
# TODO : tej option multi-gpu
# TODO add char2idx and device args to our model when we create it

class UnconditionalHandwriting(BaseModel):
    """Class for Unconditional Handwriting generation
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_gaussian, dropout, char2idx, device):
        super(UnconditionalHandwriting, self).__init__()

        # Params
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gaussian = num_gaussian
        self.output_dim = 6 * num_gaussian + 1
        self.num_layers = num_layers
        self.dropout = dropout

        # Setting an attribute device
        # TODO faut mettre ca qq part et tej les options liées au nb de gpu..
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        #  self.norm_layer = nn.LayerNorm(hidden_dim) # TODO see if useful
        self.mixture_density_layer = nn.Linear(3 * self.hidden_dim, self.output_dim)

    def forward(self, sentences, sentences_mask, strokes, strokes_mask):
        # We don't need the sentences for the unconditional handwriting generation

        _, _, strokes, strokes_mask = sentences, sentences_mask, strokes, strokes_mask
        batch_size = strokes.size(0)

        # Initialization of the hidden layers
        hidden_1 = self.init_hidden(batch_size)
        hidden_2 = self.init_hidden(batch_size)
        hidden_3 = self.init_hidden(batch_size)

        # Forward pass

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

        # Clipping the gradients of the output of the rnn
        #nn.utils.clip_grad_value_(normalized_output_rnn, 10)  # TODO check mais ca ca sert a rien jpense

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

    def generate_sample(self, sampling_bias=1.2):
        # Prepare input sequence
        stroke = [0., 0., 0.]  # init sample
        stroke = torch.tensor(stroke).view(1, 1, 3)  # (bs, seq_len, 3)
        list_strokes = []

        # Initialization of the hidden layers
        hidden_1 = self.init_hidden(stroke.size(0))
        hidden_2 = self.init_hidden(stroke.size(0))
        hidden_3 = self.init_hidden(stroke.size(0))

        with torch.no_grad():
            for i in range(1000):  # sampling len

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
    """Class for Conditional Handwriting generation
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        """
        :param sentences:       (bs, chars_seq_len)
        :param sentences_mask:  (bs, chars_seq_len)
        :param strokes:         (bs, strokes_seq_len, 3)
        :param strokes_mask:    (bs, strokes_seq_len)
        :return: output network
        """

        batch_size = strokes.size(0)

        # Initialization of the hidden layers for the regular rnns
        hidden_2 = self.init_hidden(batch_size)
        hidden_3 = self.init_hidden(batch_size)

        # Forward pass

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

    def generate_cond_sample(self, sentence, sampling_bias=1.2):

        # Adding a space char to the sentence for computing the exit condition
        sentence += ' '

        # Transforming the sentence into a tensor
        sentence = torch.tensor(data=[self.char2idx[char] for char in sentence],
                                dtype=torch.long)

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
            for i in range(1000):  # sampling len

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
                if int(torch.argmax(phi)) == sentence.size(0):
                    break

                # # Test
                # if i // 50 == 0:
                #     print(i,' : ', int(torch.argmax(phi)))

                # Sample the next stroke
                eos = torch.bernoulli(eos)  # Decide whether to stop or continue the stroke
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


# TODO implement this class for the shared functions et ensuite conditioned and unconditioned heriteraient
#  de cette classe
class HandwritingGenerator(BaseModel):
    """Class Handwriting Generation (encapsulate unconditional and conditional handwriting)
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_gaussian_out, dropout, num_chars, num_gaussian_window):
        super(HandwritingGenerator, self).__init__()
        self.input_dim = input_dim

    def forward(self, input):
        return input
