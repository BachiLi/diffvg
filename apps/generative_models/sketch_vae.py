#!/bin/env python
"""Train a Sketch-VAE."""
import argparse
from enum import Enum
import os
import wget
import time

import numpy as np
import torch as th
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

import ttools
import ttools.interfaces
from ttools.modules import networks

import rendering
import losses
import modules
import data

import pydiffvg

LOG = ttools.get_logger(__name__)


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
OUTPUT = os.path.join(BASE_DIR, "results")


class SketchVAE(th.nn.Module):
    class ImageEncoder(th.nn.Module):
        def __init__(self, image_size=64, width=64, zdim=128):
            super(SketchVAE.ImageEncoder, self).__init__()
            self.zdim = zdim

            self.net = th.nn.Sequential(
                th.nn.Conv2d(4, width, 5, padding=2),
                th.nn.InstanceNorm2d(width),
                th.nn.ReLU(inplace=True),
                # 64x64

                th.nn.Conv2d(width, width, 5, padding=2),
                th.nn.InstanceNorm2d(width),
                th.nn.ReLU( inplace=True),
                # 64x64

                th.nn.Conv2d(width, 2*width, 5, stride=1, padding=2),
                th.nn.InstanceNorm2d(2*width),
                th.nn.ReLU( inplace=True),
                # 32x32

                th.nn.Conv2d(2*width, 2*width, 5, stride=2, padding=2),
                th.nn.InstanceNorm2d(2*width),
                th.nn.ReLU( inplace=True),
                # 16x16

                th.nn.Conv2d(2*width, 2*width, 5, stride=2, padding=2),
                th.nn.InstanceNorm2d(2*width),
                th.nn.ReLU( inplace=True),
                # 16x16

                th.nn.Conv2d(2*width, 2*width, 5, stride=2, padding=2),
                th.nn.InstanceNorm2d(2*width),
                th.nn.ReLU( inplace=True),
                # 8x8

                th.nn.Conv2d(2*width, 2*width, 5, stride=2, padding=2),
                th.nn.InstanceNorm2d(2*width),
                th.nn.ReLU( inplace=True),
                # 4x4

                modules.Flatten(),
                th.nn.Linear(4*4*2*width, 2*zdim)
            )

        def forward(self, images):
            features = self.net(images)

            # VAE params
            mu = features[:, :self.zdim]
            log_sigma = features[:, self.zdim:]

            # Sample a latent vector
            sigma = th.exp(log_sigma/2.0)
            z0 = th.randn(self.zdim, device=mu.device)
            z = mu + sigma*z0

            # KL divergence needs mu/sigma
            return z, mu, log_sigma

    class ImageDecoder(th.nn.Module):
        """"""
        def __init__(self, zdim=128, image_size=64, width=64):
            super(SketchVAE.ImageDecoder, self).__init__()
            self.zdim = zdim
            self.width = width

            self.embedding = th.nn.Linear(zdim, 4*4*2*width)

            self.net = th.nn.Sequential(
                th.nn.ConvTranspose2d(2*width, 2*width, 4, padding=1, stride=2),
                th.nn.InstanceNorm2d(2*width),
                th.nn.ReLU( inplace=True),
                # 8x8

                th.nn.ConvTranspose2d(2*width, 2*width, 4, padding=1, stride=2),
                th.nn.InstanceNorm2d(2*width),
                th.nn.ReLU( inplace=True),
                # 16x16

                th.nn.ConvTranspose2d(2*width, 2*width, 4, padding=1, stride=2),
                th.nn.InstanceNorm2d(2*width),
                th.nn.ReLU( inplace=True),
                # 16x16

                th.nn.Conv2d(2*width, 2*width, 5, padding=2, stride=1),
                th.nn.InstanceNorm2d(2*width),
                th.nn.ReLU( inplace=True),
                # 16x16

                th.nn.ConvTranspose2d(2*width, 2*width, 4, padding=1, stride=2),
                th.nn.InstanceNorm2d(2*width),
                th.nn.ReLU( inplace=True),
                # 32x32

                th.nn.Conv2d(2*width, width, 5, padding=2, stride=1),
                th.nn.InstanceNorm2d(width),
                th.nn.ReLU( inplace=True),
                # 32x32

                th.nn.ConvTranspose2d(width, width, 5, padding=2, stride=1),
                th.nn.InstanceNorm2d(width),
                th.nn.ReLU( inplace=True),
                # 64x64

                th.nn.Conv2d(width, width, 5, padding=2, stride=1),
                th.nn.InstanceNorm2d(width),
                th.nn.ReLU( inplace=True),
                # 64x64

                th.nn.Conv2d(width, 4, 5, padding=2, stride=1),
            )

        def forward(self, z):
            bs = z.shape[0]
            im = self.embedding(z).view(bs, 2*self.width, 4, 4)
            out = self.net(im)
            return out

    class SketchDecoder(th.nn.Module):
        """
        The decoder outputs a sequence where each time step models (dx, dy,
        opacity).
        """
        def __init__(self, sequence_length, hidden_size=512, dropout=0.9,
                     zdim=128, num_layers=3):
            super(SketchVAE.SketchDecoder, self).__init__()
            self.sequence_length = sequence_length
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.zdim = zdim

            # Maps the latent vector to an initial cell/hidden vector
            self.hidden_cell_predictor = th.nn.Linear(zdim, 2*hidden_size*num_layers)

            self.lstm = th.nn.LSTM(
                zdim, hidden_size,
                num_layers=self.num_layers, dropout=dropout,
                batch_first=True)

            self.dxdy_predictor = th.nn.Sequential(
                th.nn.Linear(hidden_size, 2),
                th.nn.Tanh(),
            )
            self.opacity_predictor = th.nn.Sequential(
                th.nn.Linear(hidden_size, 1),
                th.nn.Sigmoid(),
            )

        def forward(self, z, hidden_and_cell=None):
            # Every step in the sequence takes the latent vector as input so we 
            # replicate it here
            bs = z.shape[0]
            steps = self.sequence_length - 1  # no need to predict the start of sequence
            expanded_z = z.unsqueeze(1).repeat(1, steps, 1)

            if hidden_and_cell is None:
                # Initialize from latent vector
                hidden_and_cell = self.hidden_cell_predictor(
                    th.tanh(z))
                hidden = hidden_and_cell[:, :self.hidden_size*self.num_layers]
                hidden = hidden.view(-1, self.num_layers, self.hidden_size)
                hidden = hidden.permute(1, 0, 2).contiguous()
                # hidden = hidden.unsqueeze(1).contiguous()
                cell = hidden_and_cell[:, self.hidden_size*self.num_layers:]
                cell = cell.view(-1, self.num_layers, self.hidden_size)
                cell = cell.permute(1, 0, 2).contiguous()
                # cell = cell.unsqueeze(1).contiguous()
                hidden_and_cell = (hidden, cell)

            outputs, hidden_and_cell = self.lstm(expanded_z, hidden_and_cell)
            hidden, cell = hidden_and_cell

            dxdy = self.dxdy_predictor(
                outputs.reshape(bs*steps, self.hidden_size)).view(bs, steps, -1)

            opacity = self.opacity_predictor(
                outputs.reshape(bs*steps, self.hidden_size)).view(bs, steps, -1)

            strokes = th.cat([dxdy, opacity], -1)

            return strokes

    def __init__(self, sequence_length, zdim=128, image_size=64):
        super(SketchVAE, self).__init__()
        self.im_encoder = SketchVAE.ImageEncoder(
            zdim=zdim, image_size=image_size)
        self.im_decoder = SketchVAE.ImageDecoder(
            zdim=zdim, image_size=image_size)
        self.sketch_decoder = SketchVAE.SketchDecoder(
            sequence_length, zdim=zdim)

    def forward(self, images):
        # Encode the images as latent vectors
        z, mu, log_sigma = self.im_encoder(images)
        decoded_im = self.im_decoder(z)
        decoded_sketch = self.sketch_decoder(z)

        return {
            "decoded_im": decoded_im,
            "decoded_sketch": decoded_sketch,
            "z": z,
            "mu": mu,
            "log_sigma": log_sigma,
        }


class SketchVAECallback(ttools.callbacks.ImageDisplayCallback):
    """Simple callback that visualize images."""
    def visualized_image(self, batch, step_data, is_val=False):
        if is_val:
            return None

        # only display the first n drawings
        n = 8
        gt = step_data["gt_image"][:n].detach()
        vae_im = step_data["vae_image"][:n].detach()
        sketch_im = step_data["sketch_image"][:n].detach()

        rendering = th.cat([gt, vae_im, sketch_im], 2)
        rendering = th.clamp(rendering, 0, 1)
        alpha =  rendering[:, 3:4]
        rendering = rendering[:, :3] * alpha

        return rendering

    def caption(self, batch, step_data, is_val=False):
        if is_val:
            return ""
        else:
            return "top: truth, middle: vae sample, output: rnn-output"




class Interface(ttools.ModelInterface):
    def __init__(self, model, lr=1e-4, lr_decay=0.9999,
                 kl_weight=0.5, kl_min_weight=0.01, kl_decay=0.99995,
                 raster_resolution=64, absolute_coords=False,
                 device="cpu", grad_clip=1.0):
        super(Interface, self).__init__()

        self.grad_clip = grad_clip
        self.raster_resolution = raster_resolution
        self.absolute_coords = absolute_coords

        self.model = model
        self.device = device
        self.model.to(self.device)
        self.im_enc_opt = th.optim.Adam(
            self.model.im_encoder.parameters(), lr=lr)
        self.im_dec_opt = th.optim.Adam(
            self.model.im_decoder.parameters(), lr=lr)
        self.sketch_dec_opt = th.optim.Adam(
            self.model.sketch_decoder.parameters(), lr=lr)

        self.kl_weight = kl_weight
        self.kl_min_weight = kl_min_weight
        self.kl_decay = kl_decay
        self.kl_loss = losses.KLDivergence()

        self.schedulers = [
            th.optim.lr_scheduler.ExponentialLR(self.im_enc_opt, lr_decay),
            th.optim.lr_scheduler.ExponentialLR(self.im_dec_opt, lr_decay),
            th.optim.lr_scheduler.ExponentialLR(self.sketch_dec_opt, lr_decay),
        ]

        # include loss on alpha
        self.im_loss = losses.MultiscaleMSELoss(channels=4).to(self.device)

    def optimizers(self):
        return [self.im_enc_opt, self.im_dec_opt, self.sketch_dec_opt]

    def kl_scaling(self):
        # Scale the KL divergence weight
        try:
            state = self.im_enc_opt.state_dict()["param_groups"][0]["params"][0]
            optim_step = self.im_enc_opt.state_dict()["state"][state]["step"]
        except KeyError:
            optim_step = 0  # no step taken yet
        kl_scaling = 1.0 - (1.0 -
                            self.kl_min_weight)*(self.kl_decay**optim_step)
        return kl_scaling

    def training_step(self, batch):
        gt_strokes, gt_im = batch
        gt_strokes = gt_strokes.to(self.device)
        gt_im = gt_im.to(self.device)

        out = self.model(gt_im)

        kl_loss = self.kl_loss(
            out["mu"], out["log_sigma"])
        kl_weight = self.kl_weight * self.kl_scaling()

        # add start of sequence
        sos = gt_strokes[:, :1]
        sketch = th.cat([sos, out["decoded_sketch"]], 1)

        vae_im = out["decoded_im"]

        # start = time.time()
        sketch_im = rendering.opacityStroke2diffvg(
            sketch, canvas_size=self.raster_resolution, debug=False,
            force_cpu=True, relative=not self.absolute_coords)
        # elapsed = (time.time() - start)*1000
        # print("out rendering took %.2fms" % elapsed)

        vae_im_loss = self.im_loss(vae_im, gt_im)
        sketch_im_loss = self.im_loss(sketch_im, gt_im)

        # vae_im_loss = th.nn.functional.mse_loss(vae_im, gt_im)
        # sketch_im_loss = th.nn.functional.mse_loss(sketch_im, gt_im)

        loss = vae_im_loss + kl_loss*kl_weight + sketch_im_loss

        self.im_enc_opt.zero_grad()
        self.im_dec_opt.zero_grad()
        self.sketch_dec_opt.zero_grad()
        loss.backward()

        # clip gradients
        enc_nrm = th.nn.utils.clip_grad_norm_(
            self.model.im_encoder.parameters(), self.grad_clip)
        dec_nrm = th.nn.utils.clip_grad_norm_(
            self.model.im_decoder.parameters(), self.grad_clip)
        sketch_dec_nrm = th.nn.utils.clip_grad_norm_(
            self.model.sketch_decoder.parameters(), self.grad_clip)

        if enc_nrm > self.grad_clip:
            LOG.debug("Clipped encoder gradient (%.5f) to %.2f",
                      enc_nrm, self.grad_clip)

        if dec_nrm > self.grad_clip:
            LOG.debug("Clipped decoder gradient (%.5f) to %.2f",
                      dec_nrm, self.grad_clip)

        if sketch_dec_nrm > self.grad_clip:
            LOG.debug("Clipped sketch decoder gradient (%.5f) to %.2f",
                      sketch_dec_nrm, self.grad_clip)

        self.im_enc_opt.step()
        self.im_dec_opt.step()
        self.sketch_dec_opt.step()

        return {
            "vae_image": vae_im,
            "sketch_image": sketch_im,
            "gt_image": gt_im,
            "loss": loss.item(),
            "vae_im_loss": vae_im_loss.item(),
            "sketch_im_loss": sketch_im_loss.item(),
            "kl_loss": kl_loss.item(),
            "kl_weight": kl_weight,
            "lr": self.im_enc_opt.param_groups[0]["lr"],
        }

    def init_validation(self):
        return dict(sample=None)

    def validation_step(self, batch, running_data):
        # Switch to eval mode for dropout, batchnorm, etc
        # self.model.eval()
        # with th.no_grad():
        #     # sample = self.model.sample(
        #     #     batch.to(self.device), temperature=self.sampling_temperature)
        #     # running_data["sample"] = sample
        # self.model.train()
        return running_data


def train(args):
    th.manual_seed(0)
    np.random.seed(0)

    dataset = data.FixedLengthQuickDrawDataset(
        args.dataset, max_seq_length=args.sequence_length,
        canvas_size=args.raster_resolution)
    dataloader = DataLoader(
        dataset, batch_size=args.bs, num_workers=args.workers, shuffle=True)

    # val_dataset = [s for idx, s in enumerate(dataset) if idx < 8]
    # val_dataloader = DataLoader(
    #     val_dataset, batch_size=8, num_workers=4, shuffle=False)

    val_dataloader = None

    model_params = {
        "zdim": args.zdim,
        "sequence_length": args.sequence_length,
        "image_size": args.raster_resolution,
        # "encoder_dim": args.encoder_dim,
        # "decoder_dim": args.decoder_dim,
    }
    model = SketchVAE(**model_params)
    model.train()

    LOG.info("Model parameters:\n%s", model_params)

    device = "cpu"
    if th.cuda.is_available():
        device = "cuda"
        LOG.info("Using CUDA")

    interface = Interface(model, raster_resolution=args.raster_resolution,
                          lr=args.lr, lr_decay=args.lr_decay,
                          kl_decay=args.kl_decay, kl_weight=args.kl_weight,
                          absolute_coords=args.absolute_coordinates,
                          device=device)

    env_name = "sketch_vae"
    if args.custom_name is not None:
        env_name += "_" + args.custom_name

    if args.absolute_coordinates:
        env_name += "_abs_coords"

    chkpt = os.path.join(OUTPUT, env_name)

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
    losses = ["loss", "kl_loss", "vae_im_loss", "sketch_im_loss"]
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

    trainer.add_callback(SketchVAECallback(
        env=env_name, win="samples", port=args.port, frequency=args.freq))

    # Start training
    trainer.train(dataloader, starting_epoch=epoch,
                  val_dataloader=val_dataloader,
                  num_epochs=args.num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cat.npz")

    parser.add_argument("--absolute_coordinates", action="store_true",
                        default=False)

    parser.add_argument("--custom_name")

    # Training params
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=0.9999)
    parser.add_argument("--kl_weight", type=float, default=0.5)
    parser.add_argument("--kl_decay", type=float, default=0.99995)

    # Model configuration
    parser.add_argument("--zdim", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=50)
    parser.add_argument("--raster_resolution", type=int, default=64)
    # parser.add_argument("--encoder_dim", type=int, default=256)
    # parser.add_argument("--decoder_dim", type=int, default=512)

    # Viz params
    parser.add_argument("--freq", type=int, default=10)
    parser.add_argument("--port", type=int, default=5000)

    args = parser.parse_args()

    pydiffvg.set_use_gpu(False)

    train(args)
