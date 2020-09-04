#!/bin/env python
"""Train a Sketch-RNN."""
import argparse
from enum import Enum
import os
import wget

import numpy as np
import torch as th
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

import ttools
import ttools.interfaces
from ttools.modules import networks

import pydiffvg

import rendering
import losses
import data

LOG = ttools.get_logger(__name__)


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
OUTPUT = os.path.join(BASE_DIR, "results", "sketch_rnn_diffvg")
OUTPUT_BASELINE = os.path.join(BASE_DIR, "results", "sketch_rnn")


class SketchRNN(th.nn.Module):
    class Encoder(th.nn.Module):
        def __init__(self, hidden_size=512, dropout=0.9, zdim=128,
                     num_layers=1):
            super(SketchRNN.Encoder, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.zdim = zdim

            self.lstm = th.nn.LSTM(5, hidden_size, num_layers=self.num_layers,
                                   dropout=dropout, bidirectional=True,
                                   batch_first=True)

            # bidirectional model -> *2
            self.mu_predictor = th.nn.Linear(2*hidden_size, zdim)
            self.sigma_predictor = th.nn.Linear(2*hidden_size, zdim)

        def forward(self, sequences, hidden_and_cell=None):
            bs = sequences.shape[0]
            if hidden_and_cell is None:
                hidden = th.zeros(self.num_layers*2, bs, self.hidden_size).to(
                    sequences.device)
                cell = th.zeros(self.num_layers*2, bs, self.hidden_size).to(
                    sequences.device)
                hidden_and_cell = (hidden, cell)

            out, hidden_and_cell = self.lstm(sequences, hidden_and_cell)
            hidden = hidden_and_cell[0]

            # Concat the forward/backward states
            fc_input = th.cat([hidden[0], hidden[1]], 1)

            # VAE params
            mu = self.mu_predictor(fc_input)
            log_sigma = self.sigma_predictor(fc_input)

            # Sample a latent vector
            sigma = th.exp(log_sigma/2.0)
            z0 = th.randn(self.zdim, device=mu.device)
            z = mu + sigma*z0

            # KL divergence needs mu/sigma
            return z, mu, log_sigma

    class Decoder(th.nn.Module):
        """
        The decoder outputs a sequence where each time step models (dx, dy) as
        a mixture of `num_gaussians` 2D Gaussians and the state triplet is a
        categorical distribution.

        The model outputs at each time step:
            - 5 parameters for each Gaussian: mu_x, mu_y, sigma_x, sigma_y,
              rho_xy
            - 1 logit for each Gaussian (the mixture weight)
            - 3 logits for the state triplet probabilities
        """
        def __init__(self, hidden_size=512, dropout=0.9, zdim=128,
                     num_layers=1, num_gaussians=20):
            super(SketchRNN.Decoder, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.zdim = zdim
            self.num_gaussians = num_gaussians

            # Maps the latent vector to an initial cell/hidden vector
            self.hidden_cell_predictor = th.nn.Linear(zdim, 2*hidden_size)

            self.lstm = th.nn.LSTM(
                5 + zdim, hidden_size,
                num_layers=self.num_layers, dropout=dropout,
                batch_first=True)

            self.parameters_predictor = th.nn.Linear(
                hidden_size, num_gaussians + 5*num_gaussians + 3)

        def forward(self, inputs, z, hidden_and_cell=None):
            # Every step in the sequence takes the latent vector as input so we
            # replicate it here
            expanded_z = z.unsqueeze(1).repeat(1, inputs.shape[1], 1)
            inputs = th.cat([inputs, expanded_z], 2)

            bs, steps = inputs.shape[:2]
            if hidden_and_cell is None:
                # Initialize from latent vector
                hidden_and_cell = self.hidden_cell_predictor(th.tanh(z))
                hidden = hidden_and_cell[:, :self.hidden_size]
                hidden = hidden.unsqueeze(0).contiguous()
                cell = hidden_and_cell[:, self.hidden_size:]
                cell = cell.unsqueeze(0).contiguous()
                hidden_and_cell = (hidden, cell)

            outputs, hidden_and_cell = self.lstm(inputs, hidden_and_cell)
            hidden, cell = hidden_and_cell

            # if self.training:
            # At train time we want parameters for each time step
            outputs = outputs.reshape(bs*steps, self.hidden_size)
            params = self.parameters_predictor(outputs).view(bs, steps, -1)

            pen_logits = params[..., -3:]
            gaussian_params = params[..., :-3]
            mixture_logits = gaussian_params[..., :self.num_gaussians]
            gaussian_params = gaussian_params[..., self.num_gaussians:].view(
                bs, steps, self.num_gaussians, -1)

            return pen_logits, mixture_logits, gaussian_params, hidden_and_cell

    def __init__(self, zdim=128, num_gaussians=20, encoder_dim=256,
                 decoder_dim=512):
        super(SketchRNN, self).__init__()
        self.encoder = SketchRNN.Encoder(zdim=zdim, hidden_size=encoder_dim)
        self.decoder = SketchRNN.Decoder(zdim=zdim, hidden_size=decoder_dim,
                                         num_gaussians=num_gaussians)

    def forward(self, sequences):
        # Encode the sequences as latent vectors
        # We skip the first time step since it is the same for all sequences:
        # (0, 0, 1, 0, 0)
        z, mu, log_sigma = self.encoder(sequences[:, 1:])

        # Decode the latent vector into a model sequence
        # Do not process the last time step (it is an end-of-sequence token)
        pen_logits, mixture_logits, gaussian_params, hidden_and_cell = \
            self.decoder(sequences[:, :-1], z)

        return {
            "pen_logits": pen_logits,
            "mixture_logits": mixture_logits,
            "gaussian_params": gaussian_params,
            "z": z,
            "mu": mu,
            "log_sigma": log_sigma,
            "hidden_and_cell": hidden_and_cell,
        }

    def sample(self, sequences, temperature=1.0):
        # Compute a latent vector conditionned based on a real sequence
        z, _, _ = self.encoder(sequences[:, 1:])

        start_of_seq = sequences[:, :1]

        max_steps = sequences.shape[1] - 1  # last step is an end-of-seq token

        output_sequences = th.zeros_like(sequences)
        output_sequences[:, 0] = start_of_seq.squeeze(1)

        current_input = start_of_seq
        hidden_and_cell = None
        for step in range(max_steps):
            pen_logits, mixture_logits, gaussian_params, hidden_and_cell = \
                self.decoder(current_input, z, hidden_and_cell=hidden_and_cell)

            # Pen and displacement state for the next step
            next_state = th.zeros_like(current_input)

            # Adjust temperature to control randomness
            mixture_logits = mixture_logits*temperature
            pen_logits = pen_logits*temperature

            # Select one of 3 pen states
            pen_distrib = \
                th.distributions.categorical.Categorical(logits=pen_logits)
            pen_state = pen_distrib.sample()

            # One-hot encoding of the state
            next_state[:, :, 2:].scatter_(2, pen_state.unsqueeze(-1),
                                          th.ones_like(next_state[:, :, 2:]))

            # Select one of the Gaussians from the mixture
            mixture_distrib = \
                th.distributions.categorical.Categorical(logits=mixture_logits)
            mixture_idx = mixture_distrib.sample()

            # select the Gaussian parameter
            mixture_idx = mixture_idx.unsqueeze(-1).unsqueeze(-1)
            mixture_idx = mixture_idx.repeat(1, 1, 1, 5)
            params = th.gather(gaussian_params, 2, mixture_idx).squeeze(2)

            # Sample a Gaussian from the corresponding Gaussian
            mu = params[..., :2]
            sigma_x = params[..., 2].exp()
            sigma_y = params[..., 3].exp()
            rho_xy = th.tanh(params[..., 4])
            cov = th.zeros(params.shape[0], params.shape[1], 2, 2,
                           device=params.device)
            cov[..., 0, 0] = sigma_x.pow(2)*temperature
            cov[..., 1, 1] = sigma_x.pow(2)*temperature
            cov[..., 1, 0] = sigma_x*sigma_y*rho_xy*temperature
            point_distrib = \
                th.distributions.multivariate_normal.MultivariateNormal(
                    mu, scale_tril=cov)
            point = point_distrib.sample()
            next_state[:, :, :2] = point

            # Commit step to output
            output_sequences[:, step + 1] = next_state.squeeze(1)

            # Prepare next recurrent step
            current_input = next_state

        return output_sequences


class SketchRNNCallback(ttools.callbacks.ImageDisplayCallback):
    """Simple callback that visualize images."""
    def visualized_image(self, batch, step_data, is_val=False):
        if not is_val:
            # No need to render training data
            return None

        with th.no_grad():
            # only display the first n drawings
            n = 8
            batch = batch[:n]

            out_im = rendering.stroke2diffvg(step_data["sample"][:n])
            im = rendering.stroke2diffvg(batch)
            im = th.cat([im, out_im], 2)

        return im

    def caption(self, batch, step_data, is_val=False):
        if is_val:
            return "top: truth, bottom: sample"
        else:
            return "top: truth, bottom: sample"


class Interface(ttools.ModelInterface):
    def __init__(self, model, lr=1e-3, lr_decay=0.9999,
                 kl_weight=0.5, kl_min_weight=0.01, kl_decay=0.99995,
                 device="cpu", grad_clip=1.0, sampling_temperature=0.4):
        super(Interface, self).__init__()
        self.grad_clip = grad_clip
        self.sampling_temperature = sampling_temperature

        self.model = model
        self.device = device
        self.model.to(self.device)
        self.enc_opt = th.optim.Adam(self.model.encoder.parameters(), lr=lr)
        self.dec_opt = th.optim.Adam(self.model.decoder.parameters(), lr=lr)

        self.kl_weight = kl_weight
        self.kl_min_weight = kl_min_weight
        self.kl_decay = kl_decay
        self.kl_loss = losses.KLDivergence()

        self.schedulers = [
            th.optim.lr_scheduler.ExponentialLR(self.enc_opt, lr_decay),
            th.optim.lr_scheduler.ExponentialLR(self.dec_opt, lr_decay),
        ]

        self.reconstruction_loss = losses.GaussianMixtureReconstructionLoss()

    def optimizers(self):
        return [self.enc_opt, self.dec_opt]

    def training_step(self, batch):
        batch = batch.to(self.device)
        out = self.model(batch)

        kl_loss = self.kl_loss(
            out["mu"], out["log_sigma"])

        # The target to predict is the next sequence step
        targets = batch[:, 1:].to(self.device)

        # Scale the KL divergence weight
        try:
            state = self.enc_opt.state_dict()["param_groups"][0]["params"][0]
            optim_step = self.enc_opt.state_dict()["state"][state]["step"]
        except KeyError:
            optim_step = 0  # no step taken yet
        kl_scaling = 1.0 - (1.0 -
                            self.kl_min_weight)*(self.kl_decay**optim_step)
        kl_weight = self.kl_weight * kl_scaling

        reconstruction_loss = self.reconstruction_loss(
            out["pen_logits"], out["mixture_logits"],
            out["gaussian_params"], targets)
        loss = kl_loss*self.kl_weight + reconstruction_loss

        self.enc_opt.zero_grad()
        self.dec_opt.zero_grad()
        loss.backward()

        # clip gradients
        enc_nrm = th.nn.utils.clip_grad_norm_(
            self.model.encoder.parameters(), self.grad_clip)
        dec_nrm = th.nn.utils.clip_grad_norm_(
            self.model.decoder.parameters(), self.grad_clip)

        if enc_nrm > self.grad_clip:
            LOG.debug("Clipped encoder gradient (%.5f) to %.2f",
                      enc_nrm, self.grad_clip)

        if dec_nrm > self.grad_clip:
            LOG.debug("Clipped decoder gradient (%.5f) to %.2f",
                      dec_nrm, self.grad_clip)

        self.enc_opt.step()
        self.dec_opt.step()

        return {
            "loss": loss.item(),
            "kl_loss": kl_loss.item(),
            "kl_weight": kl_weight,
            "recons_loss": reconstruction_loss.item(),
            "lr": self.enc_opt.param_groups[0]["lr"],
        }

    def init_validation(self):
        return dict(sample=None)

    def validation_step(self, batch, running_data):
        # Switch to eval mode for dropout, batchnorm, etc
        self.model.eval()
        with th.no_grad():
            sample = self.model.sample(
                batch.to(self.device), temperature=self.sampling_temperature)
            running_data["sample"] = sample
        self.model.train()
        return running_data


def train(args):
    th.manual_seed(0)
    np.random.seed(0)

    dataset = data.QuickDrawDataset(args.dataset)
    dataloader = DataLoader(
        dataset, batch_size=args.bs, num_workers=4, shuffle=True,
        pin_memory=False)

    val_dataset = [s for idx, s in enumerate(dataset) if idx < 8]
    val_dataloader = DataLoader(
        val_dataset, batch_size=8, num_workers=4, shuffle=False,
        pin_memory=False)

    model_params = {
        "zdim": args.zdim,
        "num_gaussians": args.num_gaussians,
        "encoder_dim": args.encoder_dim,
        "decoder_dim": args.decoder_dim,
    }
    model = SketchRNN(**model_params)
    model.train()

    device = "cpu"
    if th.cuda.is_available():
        device = "cuda"
        LOG.info("Using CUDA")

    interface = Interface(model, lr=args.lr, lr_decay=args.lr_decay,
                          kl_decay=args.kl_decay, kl_weight=args.kl_weight,
                          sampling_temperature=args.sampling_temperature,
                          device=device)

    chkpt = OUTPUT_BASELINE
    env_name = "sketch_rnn"

    # Resume from checkpoint, if any
    checkpointer = ttools.Checkpointer(
        chkpt, model, meta=model_params,
        optimizers=interface.optimizers(),
        schedulers=interface.schedulers)
    extras, meta = checkpointer.load_latest()
    epoch = extras["epoch"] if extras and "epoch" in extras.keys() else 0

    if meta is not None and meta != model_params:
        LOG.info("Checkpoint's metaparams differ "
                 "from CLI, aborting: %s and %s", meta, model_params)

    trainer = ttools.Trainer(interface)

    # Add callbacks
    losses = ["loss", "kl_loss", "recons_loss"]
    training_debug = ["lr", "kl_weight"]
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(
        keys=losses, val_keys=None))
    trainer.add_callback(ttools.callbacks.VisdomLoggingCallback(
        keys=losses, val_keys=None, env=env_name, port=args.port))
    trainer.add_callback(ttools.callbacks.VisdomLoggingCallback(
        keys=training_debug, smoothing=0, val_keys=None, env=env_name,
        port=args.port))
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(
        checkpointer, max_files=2, interval=600, max_epochs=10))
    trainer.add_callback(
        ttools.callbacks.LRSchedulerCallback(interface.schedulers))

    trainer.add_callback(SketchRNNCallback(
        env=env_name, win="samples", port=args.port, frequency=args.freq))

    # Start training
    trainer.train(dataloader, starting_epoch=epoch,
                  val_dataloader=val_dataloader,
                  num_epochs=args.num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cat.npz")

    # Training params
    parser.add_argument("--bs", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=0.9999)
    parser.add_argument("--kl_weight", type=float, default=0.5)
    parser.add_argument("--kl_decay", type=float, default=0.99995)

    # Model configuration
    parser.add_argument("--zdim", type=int, default=128)
    parser.add_argument("--num_gaussians", type=int, default=20)
    parser.add_argument("--encoder_dim", type=int, default=256)
    parser.add_argument("--decoder_dim", type=int, default=512)

    parser.add_argument("--sampling_temperature", type=float, default=0.4,
                        help="controls sampling randomness. "
                        "0.0: deterministic, 1.0: unchanged")

    # Viz params
    parser.add_argument("--freq", type=int, default=100)
    parser.add_argument("--port", type=int, default=5000)

    args = parser.parse_args()

    pydiffvg.set_use_gpu(th.cuda.is_available())

    train(args)
