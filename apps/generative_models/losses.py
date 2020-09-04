"""Losses for the generative models and baselines."""
import torch as th
import numpy as np

import ttools.modules.image_operators as imops


class KLDivergence(th.nn.Module):
    """
    Args:
        min_value(float): the loss is clipped so that value below this
            number don't affect the optimization.
    """
    def __init__(self, min_value=0.2):
        super(KLDivergence, self).__init__()
        self.min_value = min_value

    def forward(self, mu, log_sigma):
        loss = -0.5 * (1.0 + log_sigma - mu.pow(2) - log_sigma.exp())
        loss = loss.mean()
        loss = th.max(loss, self.min_value*th.ones_like(loss))
        return loss


class MultiscaleMSELoss(th.nn.Module):
    def __init__(self, channels=3):
        super(MultiscaleMSELoss, self).__init__()
        self.blur = imops.GaussianBlur(1, channels=channels)

    def forward(self, im, target):
        bs, c, h, w = im.shape
        num_levels = max(int(np.ceil(np.log2(h))) - 2, 1)

        losses = []
        for lvl in range(num_levels):
            loss = th.nn.functional.mse_loss(im, target)
            losses.append(loss)
            im = th.nn.functional.interpolate(self.blur(im), 
                                              scale_factor=0.5,
                                              mode="nearest")
            target = th.nn.functional.interpolate(self.blur(target),
                                                  scale_factor=0.5,
                                                  mode="nearest")

        losses = th.stack(losses)
        return losses.sum()


def gaussian_pdfs(dx, dy, params):
    """Returns the pdf at (dx, dy) for each Gaussian in the mixture.
    """
    dx = dx.unsqueeze(-1)  # replicate dx, dy to evaluate all pdfs at once
    dy = dy.unsqueeze(-1)

    mu_x = params[..., 0]
    mu_y = params[..., 1]
    sigma_x = params[..., 2].exp()
    sigma_y = params[..., 3].exp()
    rho_xy = th.tanh(params[..., 4])

    x = ((dx-mu_x) / sigma_x).pow(2)
    y = ((dy-mu_y) / sigma_y).pow(2)

    xy = (dx-mu_x)*(dy-mu_y) / (sigma_x * sigma_y)
    arg = x + y - 2.0*rho_xy*xy
    pdf = th.exp(-arg / (2*(1.0 - rho_xy.pow(2))))
    norm = 2.0 * np.pi * sigma_x * sigma_y * (1.0 - rho_xy.pow(2)).sqrt()

    return pdf / norm


class GaussianMixtureReconstructionLoss(th.nn.Module):
    """
    Args:
    """
    def __init__(self, eps=1e-5):
        super(GaussianMixtureReconstructionLoss, self).__init__()
        self.eps = eps

    def forward(self, pen_logits, mixture_logits, gaussian_params, targets):
        dx = targets[..., 0]
        dy = targets[..., 1]
        pen_state = targets[..., 2:].argmax(-1)  # target index

        # Likelihood loss on the stroke position
        # No need to predict accurate pen position for end-of-sequence tokens
        valid_stroke = (targets[..., -1] != 1.0).float()
        mixture_weights = th.nn.functional.softmax(mixture_logits, -1)
        pdfs = gaussian_pdfs(dx, dy, gaussian_params)
        position_loss = - th.log(self.eps + (pdfs * mixture_weights).sum(-1))

        # by actual non-empty count
        position_loss = (position_loss*valid_stroke).sum() / valid_stroke.sum()

        # Classification loss for the stroke mode
        pen_loss = th.nn.functional.cross_entropy(pen_logits.view(-1, 3),
                                                  pen_state.view(-1))

        return position_loss + pen_loss
