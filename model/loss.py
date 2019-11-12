import math
import torch


def handwriting_generation_loss(gaussian_params, strokes, strokes_mask, eps=1e-6):

    gaussian_params = (param[:, :-1, :] for param in gaussian_params)  # We remove the last predicted params
    pi, mu1, mu2, sigma1, sigma2, rho, eos = gaussian_params

    # Get the target x1, x2, eos and prepare for broadcasting
    strokes_mask = strokes_mask[:, 1:].unsqueeze(-1)  # We remove the first target
    target_eos = strokes[:, 1:, 0].unsqueeze(-1)      # We remove the first target
    target_x1 = strokes[:, 1:, 1].unsqueeze(-1)       # We remove the first target
    target_x2 = strokes[:, 1:, 2].unsqueeze(-1)       # We remove the first target
    # target_x1 = target_x1.repeat(1, 1, num_gaussian)
    # target_x2 = target_x2.repeat(1, 1, num_gaussian)

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
