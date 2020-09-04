#!/bin/env python
"""Train a GAN.

Usage:

* Train a MNIST model: 

`python train_gan.py`

* Train a Quickdraw model: 

`python train_gan.py --task quickdraw`

"""
import argparse
import os

import numpy as np
import torch as th
from torch.utils.data import DataLoader

import ttools
import ttools.interfaces

import losses
import data
import models

import pydiffvg

LOG = ttools.get_logger(__name__)


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
OUTPUT = os.path.join(BASE_DIR, "results")


class Callback(ttools.callbacks.ImageDisplayCallback):
    """Simple callback that visualize images."""
    def visualized_image(self, batch, step_data, is_val=False):
        if is_val:
            return

        gen = step_data["gen_image"][:16].detach()
        ref = step_data["gt_image"][:16].detach()

        # tensor to visualize, concatenate images
        vizdata = th.cat([ref, gen], 2)

        vector = step_data["vector_image"]
        if vector is not None:
            vector = vector[:16].detach()
            vizdata = th.cat([vizdata, vector], 2)

        vizdata = (vizdata + 1.0 ) * 0.5
        viz = th.clamp(vizdata, 0, 1)
        return viz

    def caption(self, batch, step_data, is_val=False):
        if step_data["vector_image"] is not None:
            s = "top: real, middle: raster, bottom: vector"
        else:
            s = "top: real, bottom: fake"
        return s


class Interface(ttools.ModelInterface):
    def __init__(self, generator, vect_generator,
                 discriminator, vect_discriminator,
                 lr=1e-4, lr_decay=0.9999,
                 gradient_penalty=10,
                 wgan_gp=False,
                 raster_resolution=32, device="cpu", grad_clip=1.0):
        super(Interface, self).__init__()

        self.wgan_gp = wgan_gp
        self.w_gradient_penalty = gradient_penalty

        self.n_critic = 1
        if self.wgan_gp:
            self.n_critic = 5

        self.grad_clip = grad_clip
        self.raster_resolution = raster_resolution

        self.gen = generator
        self.vect_gen = vect_generator
        self.discrim = discriminator
        self.vect_discrim = vect_discriminator

        self.device = device
        self.gen.to(self.device)
        self.discrim.to(self.device)

        beta1 = 0.5
        beta2 = 0.9

        self.gen_opt = th.optim.Adam(
            self.gen.parameters(), lr=lr, betas=(beta1, beta2))
        self.discrim_opt = th.optim.Adam(
            self.discrim.parameters(), lr=lr, betas=(beta1, beta2))

        self.schedulers = [
            th.optim.lr_scheduler.ExponentialLR(self.gen_opt, lr_decay),
            th.optim.lr_scheduler.ExponentialLR(self.discrim_opt, lr_decay),
        ]

        self.optimizers = [self.gen_opt, self.discrim_opt]

        if self.vect_gen is not None:
            assert self.vect_discrim is not None

            self.vect_gen.to(self.device)
            self.vect_discrim.to(self.device)

            self.vect_gen_opt = th.optim.Adam(
                self.vect_gen.parameters(), lr=lr, betas=(beta1, beta2))
            self.vect_discrim_opt = th.optim.Adam(
                self.vect_discrim.parameters(), lr=lr, betas=(beta1, beta2))

            self.schedulers += [
                th.optim.lr_scheduler.ExponentialLR(self.vect_gen_opt,
                                                    lr_decay),
                th.optim.lr_scheduler.ExponentialLR(self.vect_discrim_opt,
                                                    lr_decay),
            ]

            self.optimizers += [self.vect_gen_opt, self.vect_discrim_opt]

        # include loss on alpha
        self.im_loss = losses.MultiscaleMSELoss(channels=4).to(self.device)

        self.iter = 0
        
        self.cross_entropy = th.nn.BCEWithLogitsLoss()
        self.mse = th.nn.MSELoss()

    def _gradient_penalty(self, discrim, fake, real):
        bs = real.size(0)
        epsilon = th.rand(bs, 1, 1, 1, device=real.device)
        epsilon = epsilon.expand_as(real)

        interpolation = epsilon * real.data + (1 - epsilon) * fake.data
        interpolation = th.autograd.Variable(interpolation, requires_grad=True)

        interpolation_logits = discrim(interpolation)
        grad_outputs = th.ones(interpolation_logits.size(), device=real.device)

        gradients = th.autograd.grad(outputs=interpolation_logits,
                                     inputs=interpolation,
                                     grad_outputs=grad_outputs,
                                     create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(bs, -1)
        gradients_norm = th.sqrt(th.sum(gradients ** 2, dim=1) + 1e-12)

        # [Tanh-Tung 2019] https://openreview.net/pdf?id=ByxPYjC5KQ
        return self.w_gradient_penalty * ((gradients_norm - 0) ** 2).mean()

        # return self.w_gradient_penalty * ((gradients_norm - 1) ** 2).mean()

    def _discriminator_step(self, discrim, opt, fake, real):
        """Try to classify fake as 0 and real as 1."""

        opt.zero_grad()

        # no backprop to gen
        fake = fake.detach()

        fake_pred = discrim(fake)
        real_pred = discrim(real)

        if self.wgan_gp:
            gradient_penalty = self._gradient_penalty(discrim, fake, real)
            loss_d = fake_pred.mean() - real_pred.mean() + gradient_penalty
            gradient_penalty = gradient_penalty.item()
        else:
            fake_loss = self.cross_entropy(fake_pred, th.zeros_like(fake_pred))
            real_loss = self.cross_entropy(real_pred, th.ones_like(real_pred))
            # fake_loss = self.mse(fake_pred, th.zeros_like(fake_pred))
            # real_loss = self.mse(real_pred, th.ones_like(real_pred))
            loss_d = 0.5*(fake_loss + real_loss)
            gradient_penalty = None

        loss_d.backward()
        nrm = th.nn.utils.clip_grad_norm_(
            discrim.parameters(), self.grad_clip)
        if nrm > self.grad_clip:
            LOG.debug("Clipped discriminator gradient (%.5f) to %.2f",
                      nrm, self.grad_clip)

        opt.step()

        return loss_d.item(), gradient_penalty

    def _generator_step(self, gen, discrim, opt, fake):
        """Try to classify fake as 1."""

        opt.zero_grad()

        fake_pred = discrim(fake)

        if self.wgan_gp:
            loss_g = -fake_pred.mean()
        else:
            loss_g = self.cross_entropy(fake_pred, th.ones_like(fake_pred))
            # loss_g = self.mse(fake_pred, th.ones_like(fake_pred))

        loss_g.backward()

        # clip gradients
        nrm = th.nn.utils.clip_grad_norm_(
            gen.parameters(), self.grad_clip)
        if nrm > self.grad_clip:
            LOG.debug("Clipped generator gradient (%.5f) to %.2f",
                      nrm, self.grad_clip)

        opt.step()

        return loss_g.item()

    def training_step(self, batch):
        im = batch
        im = im.to(self.device)

        z = self.gen.sample_z(im.shape[0], device=self.device)

        generated = self.gen(z)

        vect_generated = None
        if self.vect_gen is not None:
            vect_generated = self.vect_gen(z)

        loss_g = None
        loss_d = None
        loss_g_vect = None
        loss_d_vect = None

        gp = None
        gp_vect = None

        if self.iter < self.n_critic:  # Discriminator update
            self.iter += 1

            loss_d, gp = self._discriminator_step(
                self.discrim, self.discrim_opt, generated, im)

            if vect_generated is not None:
                loss_d_vect, gp_vect = self._discriminator_step(
                    self.vect_discrim, self.vect_discrim_opt, vect_generated, im)

        else:  # Generator update
            self.iter = 0

            loss_g = self._generator_step(
                self.gen, self.discrim, self.gen_opt, generated)

            if vect_generated is not None:
                loss_g_vect = self._generator_step(
                    self.vect_gen, self.vect_discrim, self.vect_gen_opt, vect_generated)

        return {
            "loss_g": loss_g,
            "loss_d": loss_d,
            "loss_g_vect": loss_g_vect,
            "loss_d_vect": loss_d_vect,
            "gp": gp,
            "gp_vect": gp_vect,
            "gt_image": im,
            "gen_image": generated,
            "vector_image": vect_generated,
            "lr": self.gen_opt.param_groups[0]["lr"],
        }

    def init_validation(self):
        return dict(sample=None)

    def validation_step(self, batch, running_data):
        # Switch to eval mode for dropout, batchnorm, etc
        self.model.eval()
        return running_data


def train(args):
    th.manual_seed(0)
    np.random.seed(0)

    color_output = False
    if args.task == "mnist":
        dataset = data.MNISTDataset(args.raster_resolution, train=True)
    elif args.task == "quickdraw":
        dataset = data.QuickDrawImageDataset(
            args.raster_resolution, train=True)
    else:
        raise NotImplementedError()

    dataloader = DataLoader(
        dataset, batch_size=args.bs, num_workers=args.workers, shuffle=True)

    val_dataloader = None

    model_params = {
        "zdim": args.zdim,
        "num_strokes": args.num_strokes,
        "imsize": args.raster_resolution,
        "stroke_width": args.stroke_width,
        "color_output": color_output,
    }
    gen = models.Generator(**model_params)
    gen.train()

    discrim = models.Discriminator(color_output=color_output)
    discrim.train()

    if args.raster_only:
        vect_gen = None
        vect_discrim = None
    else:
        if args.generator == "fc":
            vect_gen = models.VectorGenerator(**model_params)
        elif args.generator == "bezier_fc":
            vect_gen = models.BezierVectorGenerator(**model_params)
        elif args.generator in ["rnn"]:
            vect_gen = models.RNNVectorGenerator(**model_params)
        elif args.generator in ["chain_rnn"]:
            vect_gen = models.ChainRNNVectorGenerator(**model_params)
        else:
            raise NotImplementedError()
        vect_gen.train()

        vect_discrim = models.Discriminator(color_output=color_output)
        vect_discrim.train()

    LOG.info("Model parameters:\n%s", model_params)

    device = "cpu"
    if th.cuda.is_available():
        device = "cuda"
        LOG.info("Using CUDA")

    interface = Interface(gen, vect_gen, discrim, vect_discrim,
                          raster_resolution=args.raster_resolution, lr=args.lr,
                          wgan_gp=args.wgan_gp,
                          lr_decay=args.lr_decay, device=device)

    env_name = args.task + "_gan"

    if args.raster_only:
        env_name += "_raster"
    else:
        env_name += "_vector"

    env_name += "_" + args.generator

    if args.wgan_gp:
        env_name += "_wgan"

    chkpt = os.path.join(OUTPUT, env_name)

    meta = {
        "model_params": model_params,
        "task": args.task,
        "generator": args.generator,
    }
    checkpointer = ttools.Checkpointer(
        chkpt, gen, meta=meta,
        optimizers=interface.optimizers,
        schedulers=interface.schedulers,
        prefix="g_")
    checkpointer_d = ttools.Checkpointer(
        chkpt, discrim, 
        prefix="d_")

    # Resume from checkpoint, if any
    extras, _ = checkpointer.load_latest()
    checkpointer_d.load_latest()

    if not args.raster_only:
        checkpointer_vect = ttools.Checkpointer(
            chkpt, vect_gen, meta=meta,
            optimizers=interface.optimizers,
            schedulers=interface.schedulers,
            prefix="vect_g_")
        checkpointer_d_vect = ttools.Checkpointer(
            chkpt, vect_discrim, 
            prefix="vect_d_")
        extras, _ = checkpointer_vect.load_latest()
        checkpointer_d_vect.load_latest()

    epoch = extras["epoch"] if extras and "epoch" in extras.keys() else 0

    # if meta is not None and meta["model_parameters"] != model_params:
    #     LOG.info("Checkpoint's metaparams differ "
    #              "from CLI, aborting: %s and %s", meta, model_params)

    trainer = ttools.Trainer(interface)

    # Add callbacks
    losses = ["loss_g", "loss_d", "loss_g_vect", "loss_d_vect", "gp",
              "gp_vect"]
    training_debug = ["lr"]

    trainer.add_callback(Callback(
        env=env_name, win="samples", port=args.port, frequency=args.freq))
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(
        keys=losses, val_keys=None))
    trainer.add_callback(ttools.callbacks.MultiPlotCallback(
        keys=losses, val_keys=None, env=env_name, port=args.port,
        server=args.server, base_url=args.base_url,
        win="losses", frequency=args.freq))
    trainer.add_callback(ttools.callbacks.VisdomLoggingCallback(
        keys=training_debug, smoothing=0, val_keys=None, env=env_name,
        server=args.server, base_url=args.base_url,
        port=args.port))
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(
        checkpointer, max_files=2, interval=600, max_epochs=10))
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(
        checkpointer_d, max_files=2, interval=600, max_epochs=10))

    if not args.raster_only:
        trainer.add_callback(ttools.callbacks.CheckpointingCallback(
            checkpointer_vect, max_files=2, interval=600, max_epochs=10))
        trainer.add_callback(ttools.callbacks.CheckpointingCallback(
            checkpointer_d_vect, max_files=2, interval=600, max_epochs=10))

    trainer.add_callback(
        ttools.callbacks.LRSchedulerCallback(interface.schedulers))

    # Start training
    trainer.train(dataloader, starting_epoch=epoch,
                  val_dataloader=val_dataloader,
                  num_epochs=args.num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        default="mnist",
                        choices=["mnist", "quickdraw"])
    parser.add_argument("--generator", 
                        default="bezier_fc",
                        choices=["bezier_fc", "fc", "rnn", "chain_rnn"],
                        help="model to use as generator")

    parser.add_argument("--raster_only", action="store_true", default=False,
                        help="if true only train the raster baseline")

    parser.add_argument("--standard_gan", dest="wgan_gp", action="store_false",
                        default=True,
                        help="if true, use regular GAN instead of WGAN")

    # Training params
    parser.add_argument("--bs", type=int, default=4, help="batch size")
    parser.add_argument("--workers", type=int, default=4,
                        help="number of dataloader threads")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.9999,
                        help="exponential learning rate decay rate")

    # Model configuration
    parser.add_argument("--zdim", type=int, default=32,
                        help="latent space dimension")
    parser.add_argument("--stroke_width", type=float, nargs=2,
                        default=(0.5, 1.5),
                        help="min and max stroke width")
    parser.add_argument("--num_strokes", type=int, default=16,
                        help="number of strokes to generate")
    parser.add_argument("--raster_resolution", type=int, default=32,
                        help="raster canvas resolution on each side")

    # Viz params
    parser.add_argument("--freq", type=int, default=10,
                        help="visualization frequency")
    parser.add_argument("--port", type=int, default=8097,
                        help="visdom port")
    parser.add_argument("--server", default=None,
                        help="visdom server if not local.")
    parser.add_argument("--base_url", default="", help="visdom entrypoint URL")

    args = parser.parse_args()

    pydiffvg.set_use_gpu(False)

    ttools.set_logger(False)

    train(args)
