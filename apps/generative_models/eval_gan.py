"""Evaluate a pretrained GAN model.
Usage:

`python eval_gan.py <path/to/model/folder>`, e.g. 
`../results/quickdraw_gan_vector_bezier_fc_wgan`.

"""
import os
import argparse
import torch as th
import numpy as np
import ttools
import imageio
from subprocess import call

import pydiffvg

import models


LOG = ttools.get_logger(__name__)


def postprocess(im, invert=False):
    im = th.clamp((im + 1.0) / 2.0, 0, 1)
    if invert:
        im = (1.0 - im)
    im = ttools.tensor2image(im)
    return im


def imsave(im, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.imwrite(path, im)


def save_scene(scn, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pydiffvg.save_svg(path, *scn, use_gamma=False)


def run(args):
    th.manual_seed(0)
    np.random.seed(0)

    meta = ttools.Checkpointer.load_meta(args.model, "vect_g_")

    if meta is None:
        LOG.warning("Could not load metadata at %s, aborting.", args.model)
        return

    LOG.info("Loaded model %s with metadata:\n %s", args.model, meta)

    if args.output_dir is None:
        outdir = os.path.join(args.model, "eval")
    else:
        outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    model_params = meta["model_params"]
    if args.imsize is not None:
        LOG.info("Overriding output image size to: %dx%d", args.imsize,
                 args.imsize)
        old_size = model_params["imsize"]
        scale = args.imsize * 1.0 / old_size
        model_params["imsize"] = args.imsize
        model_params["stroke_width"] = [w*scale for w in
                                        model_params["stroke_width"]]
        LOG.info("Overriding width to: %s", model_params["stroke_width"])

    # task = meta["task"]
    generator = meta["generator"]
    if generator == "fc":
        model = models.VectorGenerator(**model_params)
    elif generator == "bezier_fc":
        model = models.BezierVectorGenerator(**model_params)
    elif generator in ["rnn"]:
        model = models.RNNVectorGenerator(**model_params)
    elif generator in ["chain_rnn"]:
        model = models.ChainRNNVectorGenerator(**model_params)
    else:
        raise NotImplementedError()
    model.eval()

    device = "cpu"
    if th.cuda.is_available():
        device = "cuda"

    model.to(device)

    checkpointer = ttools.Checkpointer(
        args.model, model, meta=meta, prefix="vect_g_")
    checkpointer.load_latest()

    LOG.info("Computing latent space interpolation")
    for i in range(args.nsamples):
        z0 = model.sample_z(1)
        z1 = model.sample_z(1)

        # interpolation
        alpha = th.linspace(0, 1, args.nsteps).view(args.nsteps, 1).to(device)
        alpha_video = th.linspace(0, 1, args.nframes).view(args.nframes, 1)
        alpha_video = alpha_video.to(device)

        length = [args.nsteps, args.nframes]
        for idx, a in enumerate([alpha, alpha_video]):
            _z0 = z0.repeat(length[idx], 1).to(device)
            _z1 = z1.repeat(length[idx], 1).to(device)
            batch = _z0*(1-a) + _z1*a
            out = model(batch)
            if idx == 0:  # image viz
                n, c, h, w = out.shape
                out = out.permute(1, 2, 0, 3)
                out = out.contiguous().view(1, c, h, w*n)
                out = postprocess(out, invert=args.invert)
                imsave(out, os.path.join(outdir,
                                         "latent_interp", "%03d.png" % i))

                scenes = model.get_vector(batch)
                for scn_idx, scn in enumerate(scenes):
                    save_scene(scn, os.path.join(outdir, "latent_interp_svg",
                                                 "%03d" % i, "%03d.svg" %
                                                 scn_idx))
            else:  # video viz
                anim_root = os.path.join(outdir,
                                         "latent_interp_video", "%03d" % i)
                LOG.info("Rendering animation %d", i)
                for frame_idx, frame in enumerate(out):
                    LOG.info("frame %d", frame_idx)
                    frame = frame.unsqueeze(0)
                    frame = postprocess(frame, invert=args.invert)
                    imsave(frame, os.path.join(anim_root,
                                               "frame%04d.png" % frame_idx))
                call(["ffmpeg", "-framerate", "30", "-i",
                      os.path.join(anim_root, "frame%04d.png"), "-vb", "20M",
                     os.path.join(outdir,
                                  "latent_interp_video", "%03d.mp4" % i)])
        LOG.info("  saved %d", i)

    LOG.info("Sampling latent space")

    for i in range(args.nsamples):
        n = 8
        bs = n*n
        z = model.sample_z(bs).to(device)
        out = model(z)
        _, c, h, w = out.shape
        out = out.view(n, n, c, h, w).permute(2, 0, 3, 1, 4)
        out = out.contiguous().view(1, c, h*n, w*n)
        out = postprocess(out)
        imsave(out, os.path.join(outdir, "samples_%03d.png" % i))
        LOG.info("  saved %d", i)

    LOG.info("output images saved to %s", outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model")
    parser.add_argument("--output_dir", help="output directory for "
                        " the samples. Defaults to the model's path")
    parser.add_argument("--nsamples", default=16, type=int, 
                        help="number of output to compute")
    parser.add_argument("--imsize", type=int,
                        help="if provided, override the raster output "
                        "resolution")
    parser.add_argument("--nsteps", default=9, type=int, help="number of "
                        "interpolation steps for the interpolation")
    parser.add_argument("--nframes", default=120, type=int, help="number of "
                        "frames for the interpolation video")
    parser.add_argument("--invert", default=False, action="store_true",
                        help="if True, render black on white rather than the"
                        " opposite")

    args = parser.parse_args()

    pydiffvg.set_use_gpu(False)

    ttools.set_logger(False)

    run(args)
