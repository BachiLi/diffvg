#!/bin/env python
"""Train a VAE MNIST generator.

Usage:

* Train a model:

`python mnist_vae.py train`

* Generate samples from a trained model:

`python mnist_vae.py sample`

* Generate latent space interpolations from a trained model:

`python mnist_vae.py interpolate`
"""
import argparse
import os

import numpy as np
import torch as th
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

import ttools
import ttools.interfaces

from modules import Flatten

import pydiffvg

LOG = ttools.get_logger(__name__)


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
VAE_OUTPUT = os.path.join(BASE_DIR, "results", "mnist_vae")
AE_OUTPUT = os.path.join(BASE_DIR, "results", "mnist_ae")


def _onehot(label):
    bs = label.shape[0]
    label_onehot = label.new(bs, 10)
    label_onehot = label_onehot.zero_()
    label_onehot.scatter_(1, label.unsqueeze(1), 1)
    return label_onehot.float()


def render(canvas_width, canvas_height, shapes, shape_groups, samples=2):
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width,
                  canvas_height,
                  samples,
                  samples,
                  0,
                  None,
                  *scene_args)
    return img


class MNISTCallback(ttools.callbacks.ImageDisplayCallback):
    """Simple callback that visualize generated images during training."""

    def visualized_image(self, batch, step_data, is_val=False):
        im = step_data["rendering"].detach().cpu()
        im = 0.5 + 0.5*im
        ref = batch[0].cpu()

        vizdata = [im, ref]

        # tensor to visualize, concatenate images
        viz = th.clamp(th.cat(vizdata, 2), 0, 1)
        return viz

    def caption(self, batch, step_data, is_val=False):
        return "fake, real"


class VAEInterface(ttools.ModelInterface):
    def __init__(self, model, lr=1e-4, cuda=True, max_grad_norm=10,
                 variational=True, w_kld=1.0):
        super(VAEInterface, self).__init__()

        self.max_grad_norm = max_grad_norm

        self.model = model

        self.w_kld = w_kld

        self.variational = variational

        self.device = "cpu"
        if cuda:
            self.device = "cuda"

        self.model.to(self.device)

        self.opt = th.optim.Adam(
            self.model.parameters(), lr=lr, betas=(0.5, 0.5), eps=1e-12)

    def training_step(self, batch):
        im, label = batch[0], batch[1]
        im = im.to(self.device)
        label = label.to(self.device)
        rendering, auxdata = self.model(im, label)

        im = batch[0]
        im = im.to(self.device)

        logvar = auxdata["logvar"]
        mu = auxdata["mu"]

        data_loss = th.nn.functional.mse_loss(rendering, im)

        ret = {}
        if self.variational:  # VAE mode
            kld = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
            kld = kld.mean()
            loss = data_loss + kld*self.w_kld
            ret["kld"] = kld.item()
        else:  # Regular autoencoder
            loss = data_loss

        # optimize
        self.opt.zero_grad()
        loss.backward()

        # Clip large gradients if needed
        if self.max_grad_norm is not None:
            nrm = th.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm)
            if nrm > self.max_grad_norm:
                LOG.warning("Clipping generator gradients. norm = %.3f > %.3f",
                            nrm, self.max_grad_norm)

        self.opt.step()

        ret["loss"] = loss.item()
        ret["data_loss"] = data_loss.item()
        ret["auxdata"] = auxdata
        ret["rendering"] = rendering
        ret["logvar"] = logvar.abs().max().item()

        return ret


class VectorMNISTVAE(th.nn.Module):
    def __init__(self, imsize=28, paths=4, segments=5, samples=2, zdim=128,
                 conditional=False, variational=True, raster=False, fc=False,
                 stroke_width=None):
        super(VectorMNISTVAE, self).__init__()

        self.samples = samples
        self.imsize = imsize
        self.paths = paths
        self.segments = segments
        self.zdim = zdim
        self.conditional = conditional
        self.variational = variational

        if stroke_width is None:
            self.stroke_width = (1.0, 3.0)
            LOG.warning("Setting default stroke with %s", self.stroke_width)
        else:
            self.stroke_width = stroke_width

        ncond = 0
        if self.conditional:  # one hot encoded input for conditional model
            ncond = 10

        self.fc = fc
        mult = 1
        nc = 1024

        if not self.fc:  # conv model
            self.encoder = th.nn.Sequential(
                # 32x32
                th.nn.Conv2d(1 + ncond, mult*64, 4, padding=0, stride=2),
                th.nn.LeakyReLU(0.2, inplace=True),

                # 16x16
                th.nn.Conv2d(mult*64, mult*128, 4, padding=0, stride=2),
                th.nn.LeakyReLU(0.2, inplace=True),

                # 8x8
                th.nn.Conv2d(mult*128, mult*256, 4, padding=0, stride=2),
                th.nn.LeakyReLU(0.2, inplace=True),
                Flatten(),
            )
        else:
            self.encoder = th.nn.Sequential(
                # 32x32
                Flatten(),
                th.nn.Linear(28*28 + ncond, mult*256),
                th.nn.LeakyReLU(0.2, inplace=True),

                # 8x8
                th.nn.Linear(mult*256, mult*256, 4),
                th.nn.LeakyReLU(0.2, inplace=True),
            )

        self.mu_predictor = th.nn.Linear(256*1*1, zdim)
        if self.variational:
            self.logvar_predictor = th.nn.Linear(256*1*1, zdim)

        self.decoder = th.nn.Sequential(
            th.nn.Linear(zdim + ncond, nc),
            th.nn.SELU(inplace=True),

            th.nn.Linear(nc, nc),
            th.nn.SELU(inplace=True),
        )

        self.raster = raster

        if self.raster:
            self.raster_decoder = th.nn.Sequential(
                th.nn.Linear(nc, imsize*imsize),
            )
        else:
            # 4 points bezier with n_segments -> 3*n_segments + 1 points
            self.point_predictor = th.nn.Sequential(
                th.nn.Linear(nc, 2*self.paths*(self.segments*3+1)),
                th.nn.Tanh()  # bound spatial extent
            )

            self.width_predictor = th.nn.Sequential(
                th.nn.Linear(nc, self.paths),
                th.nn.Sigmoid()
            )

            self.alpha_predictor = th.nn.Sequential(
                th.nn.Linear(nc, self.paths),
                th.nn.Sigmoid()
            )

    def encode(self, im, label):
        bs, _, h, w = im.shape
        if self.conditional:
            label_onehot = _onehot(label)
            if not self.fc:
                label_onehot = label_onehot.view(
                    bs, 10, 1, 1).repeat(1, 1, h, w)
                out = self.encoder(th.cat([im, label_onehot], 1))
            else:
                out = self.encoder(th.cat([im.view(bs, -1), label_onehot], 1))
        else:
            out = self.encoder(im)
        mu = self.mu_predictor(out)
        if self.variational:
            logvar = self.logvar_predictor(out)
            return mu, logvar
        else:
            return mu

    def reparameterize(self, mu, logvar):
        std = th.exp(0.5*logvar)
        eps = th.randn_like(logvar)
        return mu + std*eps

    def _decode_features(self, z, label):
        if label is not None:
            if not self.conditional:
                raise ValueError("decoding with an input label "
                                 "requires a conditional AE")
            label_onehot = _onehot(label)
            z = th.cat([z, label_onehot], 1)

        decoded = self.decoder(z)
        return decoded

    def decode(self, z, label=None):
        bs = z.shape[0]

        feats = self._decode_features(z, label)

        if self.raster:
            out = self.raster_decoder(feats).view(
                bs, 1, self.imsize, self.imsize)
            return out, {}

        all_points = self.point_predictor(feats)
        all_points = all_points.view(bs, self.paths, -1, 2)

        all_points = all_points*(self.imsize//2-2) + self.imsize//2

        if False:
            all_widths = th.ones(bs, self.paths) * 0.5
        else:
            all_widths = self.width_predictor(feats)
            min_width = self.stroke_width[0]
            max_width = self.stroke_width[1]
            all_widths = (max_width - min_width) * all_widths + min_width

        if False:
            all_alphas = th.ones(bs, self.paths)
        else:
            all_alphas = self.alpha_predictor(feats)

        # Process the batch sequentially
        outputs = []
        scenes = []
        for k in range(bs):
            # Get point parameters from network
            shapes = []
            shape_groups = []
            for p in range(self.paths):
                points = all_points[k, p].contiguous().cpu()
                width = all_widths[k, p].cpu()
                alpha = all_alphas[k, p].cpu()

                color = th.cat([th.ones(3), alpha.view(1,)])
                num_ctrl_pts = th.zeros(self.segments, dtype=th.int32) + 2

                path = pydiffvg.Path(
                    num_control_points=num_ctrl_pts, points=points,
                    stroke_width=width, is_closed=False)

                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=th.tensor([len(shapes) - 1]),
                    fill_color=None,
                    stroke_color=color)
                shape_groups.append(path_group)

            scenes.append(
                [shapes, shape_groups, (self.imsize, self.imsize)])

            # Rasterize
            out = render(self.imsize, self.imsize, shapes, shape_groups,
                         samples=self.samples)

            # Torch format, discard alpha, make gray
            out = out.permute(2, 0, 1).view(
                4, self.imsize, self.imsize)[:3].mean(0, keepdim=True)

            outputs.append(out)

        output = th.stack(outputs).to(z.device)

        auxdata = {
            "points": all_points,
            "scenes": scenes,
        }

        # map to [-1, 1]
        output = output*2.0 - 1.0

        return output, auxdata

    def forward(self, im, label):
        if self.variational:
            mu, logvar = self.encode(im, label)
            z = self.reparameterize(mu, logvar)
        else:
            mu = self.encode(im, label)
            z = mu
            logvar = None

        if self.conditional:
            output, aux = self.decode(z, label=label)
        else:
            output, aux = self.decode(z)

        aux["logvar"] = logvar
        aux["mu"] = mu

        return output, aux


class Dataset(th.utils.data.Dataset):
    def __init__(self, data_dir, imsize):
        super(Dataset, self).__init__()
        self.mnist = dset.MNIST(root=data_dir, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]))

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        im, label = self.mnist[idx]

        # make sure data uses [0, 1] range
        im -= im.min()
        im /= im.max() + 1e-8
        im -= 0.5
        im /= 0.5
        return im, label


def train(args):
    th.manual_seed(0)
    np.random.seed(0)

    pydiffvg.set_use_gpu(args.cuda)

    # Initialize datasets
    imsize = 28
    dataset = Dataset(args.data_dir, imsize)
    dataloader = DataLoader(dataset, batch_size=args.bs,
                            num_workers=4, shuffle=True)

    if args.generator in ["vae", "ae"]:
        LOG.info("Vector config:\n  samples %d\n"
                 "  paths: %d\n  segments: %d\n"
                 "  zdim: %d\n"
                 "  conditional: %d\n"
                 "  fc: %d\n",
                 args.samples, args.paths, args.segments,
                 args.zdim, args.conditional, args.fc)

    model_params = dict(samples=args.samples, paths=args.paths,
                        segments=args.segments, conditional=args.conditional,
                        zdim=args.zdim, fc=args.fc)

    if args.generator == "vae":
        model = VectorMNISTVAE(variational=True, **model_params)
        chkpt = VAE_OUTPUT
        name = "mnist_vae"
    elif args.generator == "ae":
        model = VectorMNISTVAE(variational=False, **model_params)
        chkpt = AE_OUTPUT
        name = "mnist_ae"
    else:
        raise ValueError("unknown generator")

    if args.conditional:
        name += "_conditional"
        chkpt += "_conditional"

    if args.fc:
        name += "_fc"
        chkpt += "_fc"

    # Resume from checkpoint, if any
    checkpointer = ttools.Checkpointer(
        chkpt, model, meta=model_params, prefix="g_")
    extras, meta = checkpointer.load_latest()

    if meta is not None and meta != model_params:
        LOG.info(f"Checkpoint's metaparams differ from CLI, "
                 f"aborting: {meta} and {model_params}")

    # Hook interface
    if args.generator in ["vae", "ae"]:
        variational = args.generator == "vae"
        if variational:
            LOG.info("Using a VAE")
        else:
            LOG.info("Using an AE")
        interface = VAEInterface(model, lr=args.lr, cuda=args.cuda,
                                 variational=variational,
                                 w_kld=args.kld_weight)

    trainer = ttools.Trainer(interface)

    # Add callbacks
    keys = []
    if args.generator == "vae":
        keys = ["kld", "data_loss", "loss", "logvar"]
    elif args.generator == "ae":
        keys = ["data_loss", "loss"]
    port = 8080
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(
        keys=keys, val_keys=keys))
    trainer.add_callback(ttools.callbacks.VisdomLoggingCallback(
        keys=keys, val_keys=keys, env=name, port=port))
    trainer.add_callback(MNISTCallback(
        env=name, win="samples", port=port, frequency=args.freq))
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(
        checkpointer, max_files=2, interval=600, max_epochs=50))

    # Start training
    trainer.train(dataloader, num_epochs=args.num_epochs)


def generate_samples(args):
    chkpt = VAE_OUTPUT
    if args.conditional:
        chkpt += "_conditional"
    if args.fc:
        chkpt += "_fc"

    meta = ttools.Checkpointer.load_meta(chkpt, prefix="g_")
    if meta is None:
        LOG.info("No metadata in checkpoint (or no checkpoint), aborting.")
        return

    model = VectorMNISTVAE(**meta)
    checkpointer = ttools.Checkpointer(chkpt, model, prefix="g_")
    checkpointer.load_latest()
    model.eval()

    # Sample some latent vectors
    n = 8
    bs = n*n
    z = th.randn(bs, model.zdim)

    imsize = 28
    dataset = Dataset(args.data_dir, imsize)
    dataloader = DataLoader(dataset, batch_size=bs,
                            num_workers=1, shuffle=True)

    for batch in dataloader:
        ref, label = batch
        break

    autoencode = True
    if autoencode:
        LOG.info("Sampling with auto-encoder code")
        if not args.conditional:
            label = None
        mu, logvar = model.encode(ref, label)
        z = model.reparameterize(mu, logvar)
    else:
        label = None
        if args.conditional:
            label = th.clamp(th.rand(bs)*10, 0, 9).long()
            if args.digit is not None:
                label[:] = args.digit

    with th.no_grad():
        images, aux = model.decode(z, label=label)
        scenes = aux["scenes"]
    images += 1.0
    images /= 2.0

    h = w = model.imsize

    images = images.view(n, n, h, w).permute(0, 2, 1, 3)
    images = images.contiguous().view(n*h, n*w)
    images = th.clamp(images, 0, 1).cpu().numpy()
    path = os.path.join(chkpt, "samples.png")
    pydiffvg.imwrite(images, path, gamma=2.2)

    if autoencode:
        ref += 1.0
        ref /= 2.0
        ref = ref.view(n, n, h, w).permute(0, 2, 1, 3)
        ref = ref.contiguous().view(n*h, n*w)
        ref = th.clamp(ref, 0, 1).cpu().numpy()
        path = os.path.join(chkpt, "ref.png")
        pydiffvg.imwrite(ref, path, gamma=2.2)

    # merge scenes
    all_shapes = []
    all_shape_groups = []
    cur_id = 0
    for idx, s in enumerate(scenes):
        shapes, shape_groups, _ = s
        # width, height = sizes

        # Shift digit on canvas
        center_x = idx % n
        center_y = idx // n
        for shape in shapes:
            shape.points[:, 0] += center_x * model.imsize
            shape.points[:, 1] += center_y * model.imsize
            all_shapes.append(shape)
        for grp in shape_groups:
            grp.shape_ids[:] = cur_id
            cur_id += 1
            all_shape_groups.append(grp)

    LOG.info("Generated %d shapes", len(all_shapes))

    fname = os.path.join(chkpt, "digits.svg")
    pydiffvg.save_svg(fname, n*model.imsize, n*model.imsize, all_shapes,
                      all_shape_groups, use_gamma=False)

    LOG.info("Results saved to %s", chkpt)


def interpolate(args):
    chkpt = VAE_OUTPUT
    if args.conditional:
        chkpt += "_conditional"
    if args.fc:
        chkpt += "_fc"

    meta = ttools.Checkpointer.load_meta(chkpt, prefix="g_")
    if meta is None:
        LOG.info("No metadata in checkpoint (or no checkpoint), aborting.")
        return

    model = VectorMNISTVAE(imsize=128, **meta)
    checkpointer = ttools.Checkpointer(chkpt, model, prefix="g_")
    checkpointer.load_latest()
    model.eval()

    # Sample some latent vectors
    bs = 10
    z = th.randn(bs, model.zdim)

    label = None
    label = th.arange(0, 10)

    animation = []
    nframes = 60
    with th.no_grad():
        for idx, _z in enumerate(z):
            if idx == 0:  # skip first
                continue
            _z0 = z[idx-1].unsqueeze(0).repeat(nframes, 1)
            _z = _z.unsqueeze(0).repeat(nframes, 1)
            if args.conditional:
                _label = label[idx].unsqueeze(0).repeat(nframes)
            else:
                _label = None

            # interp weights
            alpha = th.linspace(0, 1, nframes).view(nframes,  1)
            batch = alpha*_z + (1.0 - alpha)*_z0
            images, aux = model.decode(batch, label=_label)
            images += 1.0
            images /= 2.0
            animation.append(images)

    anim_dir = os.path.join(chkpt, "interpolation")
    os.makedirs(anim_dir, exist_ok=True)
    animation = th.cat(animation, 0)
    for idx, frame in enumerate(animation):
        frame = frame.squeeze()
        frame = th.clamp(frame, 0, 1).cpu().numpy()
        path = os.path.join(anim_dir, "frame%03d.png" % idx)
        pydiffvg.imwrite(frame, path, gamma=2.2)

    LOG.info("Results saved to %s", anim_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers()

    parser.add_argument("--cpu", dest="cuda", action="store_false",
                        default=th.cuda.is_available(),
                        help="if true, use CPU instead of GPU.")
    parser.add_argument("--no-conditional", dest="conditional",
                        action="store_false", default=True)
    parser.add_argument("--no-fc", dest="fc", action="store_false",
                        default=True)
    parser.add_argument("--data_dir", default="mnist",
                        help="path to download and store the data.")

    # -- Train ----------------------------------------------------------------
    parser_train = subs.add_parser("train")
    parser_train.add_argument("--generator", choices=["vae", "ae"],
                              default="vae",
                              help="choice of regular or variational "
                              "autoencoder")
    parser_train.add_argument("--freq", type=int, default=100,
                              help="number of steps between visualizations")
    parser_train.add_argument("--lr", type=float, default=5e-5,
                              help="learning rate")
    parser_train.add_argument("--kld_weight", type=float, default=1.0,
                              help="scalar weight for the KL divergence term.")
    parser_train.add_argument("--bs", type=int, default=8, help="batch size")
    parser_train.add_argument("--num_epochs", default=50, type=int,
                              help="max number of epochs")
    # Vector configs
    parser_train.add_argument("--paths", type=int, default=1,
                              help="number of vector paths to generate.")
    parser_train.add_argument("--segments", type=int, default=3,
                              help="number of segments per vector path")
    parser_train.add_argument("--samples", type=int, default=4,
                              help="number of samples in the MC rasterizer")
    parser_train.add_argument("--zdim", type=int, default=20,
                              help="dimension of the latent space")
    parser_train.set_defaults(func=train)

    # -- Eval -----------------------------------------------------------------
    parser_sample = subs.add_parser("sample")
    parser_sample.add_argument("--digit", type=int, choices=list(range(10)),
                               help="digits to synthesize, "
                               "random if not specified")
    parser_sample.set_defaults(func=generate_samples)

    parser_interpolate = subs.add_parser("interpolate")
    parser_interpolate.set_defaults(func=interpolate)

    args = parser.parse_args()

    ttools.set_logger(True)
    args.func(args)
