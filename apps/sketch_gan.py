"""A simple training interface using ttools."""
import argparse
import os
import logging
import random

import numpy as np
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as xforms
from torch.utils.data import DataLoader

import ttools
import ttools.interfaces

import pydiffvg

LOG = ttools.get_logger(__name__)

pydiffvg.render_pytorch.print_timing = False

torch.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True

latent_dim = 100
img_size = 32
num_paths = 8
num_segments = 8

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class VisdomImageCallback(ttools.callbacks.ImageDisplayCallback):
    def visualized_image(self, batch, fwd_result):
        return torch.cat([batch[0], fwd_result.cpu()], dim = 2)

# From https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(128, 256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256, 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 1024),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, 2 * num_paths * (num_segments + 1) + num_paths + num_paths),
            torch.nn.Sigmoid()
        )

    def forward(self, z):
        out = self.fc(z)
        # construct paths
        imgs = []
        for b in range(out.shape[0]):
            index = 0
            shapes = []
            shape_groups = []
            for i in range(num_paths):
                points = img_size * out[b, index: index + 2 * (num_segments + 1)].view(-1, 2).cpu()
                index += 2 * (num_segments + 1)
                stroke_width = img_size * out[b, index].view(1).cpu()
                index += 1
                
                num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
                path = pydiffvg.Path(num_control_points = num_control_points,
                                     points = points,
                                     stroke_width = stroke_width,
                                     is_closed = False)
                shapes.append(path)
    
                stroke_color = out[b, index].view(1).cpu()
                index += 1
                stroke_color = torch.cat([stroke_color, torch.tensor([0.0, 0.0, 1.0])])
                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                                 fill_color = None,
                                                 stroke_color = stroke_color)
                shape_groups.append(path_group)
            scene_args = pydiffvg.RenderFunction.serialize_scene(img_size, img_size, shapes, shape_groups)
            render = pydiffvg.RenderFunction.apply
            img = render(img_size, # width
                         img_size, # height
                         2,   # num_samples_x
                         2,   # num_samples_y
                         random.randint(0, 1048576), # seed
                         None,
                         *scene_args)
            img = img[:, :, :1]
            # HWC -> NCHW
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
            imgs.append(img)
        img = torch.cat(imgs, dim = 0)
        return img

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [torch.nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     torch.nn.LeakyReLU(0.2, inplace=True),
                     torch.nn.Dropout2d(0.25)]
            if bn:
                block.append(torch.nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = torch.nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = torch.nn.Sequential(
            torch.nn.Linear(128 * ds_size ** 2, 1),
            torch.nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

class MNISTInterface(ttools.interfaces.SGANInterface):
    """An adapter to run or train a model."""

    def __init__(self, gen, discrim, lr=2e-4):
        super(MNISTInterface, self).__init__(gen, discrim, lr, opt = 'adam')

    def forward(self, batch):
        return self.gen(torch.zeros([batch[0].shape[0], latent_dim], device = self.device).normal_())

    def _discriminator_input(self, batch, fwd_data, fake=False):
        if fake:
            return fwd_data
        else:
            return batch[0].to(self.device)

def train(args):
    """Train a MNIST classifier."""

    # Setup train and val data
    _xform = xforms.Compose([xforms.Resize([32, 32]), xforms.ToTensor()])
    data = MNIST("data/mnist", train=True, download=True, transform=_xform)

    # Initialize asynchronous dataloaders
    loader = DataLoader(data, batch_size=args.bs, num_workers=2)

    # Instantiate the models
    gen = Generator()
    discrim = Discriminator()

    gen.apply(weights_init_normal)
    discrim.apply(weights_init_normal)

    # Checkpointer to save/recall model parameters
    checkpointer_gen = ttools.Checkpointer(os.path.join(args.out, "checkpoints"), model=gen, prefix="gen_")
    checkpointer_discrim = ttools.Checkpointer(os.path.join(args.out, "checkpoints"), model=discrim, prefix="discrim_")

    # resume from a previous checkpoint, if any
    checkpointer_gen.load_latest()
    checkpointer_discrim.load_latest()

    # Setup a training interface for the model
    interface = MNISTInterface(gen, discrim, lr=args.lr)

    # Create a training looper with the interface we defined
    trainer = ttools.Trainer(interface)

    # Adds several callbacks, that will be called by the trainer --------------
    # A periodic checkpointing operation
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(checkpointer_gen))
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(checkpointer_discrim))
    # A simple progress bar
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(
        keys=["loss_g", "loss_d", "loss"]))
    # A volatile logging using visdom
    trainer.add_callback(ttools.callbacks.VisdomLoggingCallback(
        keys=["loss_g", "loss_d", "loss"],
        port=8080, env="mnist_demo"))
    # Image
    trainer.add_callback(VisdomImageCallback(port=8080, env="mnist_demo"))
    # -------------------------------------------------------------------------

    # Start the training
    LOG.info("Training started, press Ctrl-C to interrupt.")
    trainer.train(loader, num_epochs=args.epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: subparsers
    parser.add_argument("data", help="directory where we download and store the MNIST dataset.")
    parser.add_argument("out", help="directory where we write the checkpoints and visualizations.")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for the optimizer.")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train for.")
    parser.add_argument("--bs", type=int, default=64, help="number of elements per batch.")
    args = parser.parse_args()
    ttools.set_logger(True)  # activate debug prints
    train(args)
