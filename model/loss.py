import math
import torch
import torch.nn.functional as F


def handwriting_generation_loss(gaussian_params, strokes, strokes_mask, eps=1e-6):

    gaussian_params = (param[:, :-1, :] for param in gaussian_params)  # We remove the last predicted params
    pi, mu1, mu2, sigma1, sigma2, rho, eos = gaussian_params

    # Get the target x1, x2, eos and prepare for broadcasting
    strokes_mask = strokes_mask[:, 1:].unsqueeze(-1)  # We remove the first target
    target_eos = strokes[:, 1:, 0].unsqueeze(-1)      # We remove the first target
    target_x1 = strokes[:, 1:, 1].unsqueeze(-1)       # We remove the first target
    target_x2 = strokes[:, 1:, 2].unsqueeze(-1)       # We remove the first target

    # 1) Compute gaussian loss

    # compute the pi term
    pi_term = torch.log(pi)
    # compute the sigma term
    sigma_term = -torch.log(2 * math.pi * sigma1 * sigma2 + eps)
    # compute the rho term
    rho_term = -torch.log(1 - rho ** 2 + eps) / 2.
    # compute the Z term
    Z1 = ((target_x1 - mu1) ** 2) / (sigma1 ** 2 + eps)
    Z2 = ((target_x2 - mu2) ** 2) / (sigma2 ** 2 + eps)
    Z3 = -2. * rho * (target_x1 - mu1) * (target_x2 - mu2) / (sigma1 * sigma2 + eps)
    Z = Z1 + Z2 + Z3
    Z_term = - Z / (2 * (1 - rho ** 2) + eps)
    # Compute the gaussian loss
    exponential_term = pi_term + sigma_term + rho_term + Z_term
    gaussian_loss = - torch.logsumexp(exponential_term, dim=2).unsqueeze(-1)
    gaussian_loss = (gaussian_loss * strokes_mask.float()).sum(1).mean()  # Apply the mask

    # 2) Compute the end of stroke loss

    eos_loss = - target_eos * torch.log(eos) - (1 - target_eos) * torch.log(1 - eos)
    eos_loss = (eos_loss * strokes_mask.float()).sum(1).mean()

    return gaussian_loss + eos_loss


def handwriting_recognition_loss(output_network, sentences, sentences_mask):
    # output_network shape (bs, char_seq_len, num_chars)

    # Reshape output network
    batch_size = output_network.size(0)
    num_chars = output_network.size(2)
    output_network = output_network[:, 1:].reshape(-1, num_chars)

    # Define target
    target = sentences.reshape(-1)

    # Shifting the target mask to the right -> the first 0 padding for each seq will play the role of an <eos> token
    shift_tensor = torch.tensor([[True] for i in range(batch_size)], device=sentences.device)
    sentences_mask = torch.cat([shift_tensor, sentences_mask], dim=-1)
    sentences_mask = sentences_mask[:, :-1]

    # Defining target mask
    target_mask = sentences_mask.reshape(-1).float()

    nll = F.nll_loss(output_network, target, reduction='none')
    nll = nll * target_mask  # Apply the mask
    nll = nll.sum() / batch_size

    return nll
