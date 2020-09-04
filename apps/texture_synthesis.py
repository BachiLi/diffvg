import os, sys
import pydiffvg
import argparse
import torch
# import torch as th
import scipy.ndimage.filters as filters
# import numba
import numpy as np
from skimage import io
sys.path.append('./textureSyn')
from patchBasedTextureSynthesis import *
from make_gif import make_gif
import random
import ttools.modules

from svgpathtools import svg2paths2, Path, is_path_segment
"""
python texture_synthesis.py textureSyn/traced_1.png  --svg-path textureSyn/traced_1.svg --case 1
"""

def texture_syn(img_path):
    ## get the width and height first
    # input_img = io.imread(img_path)  # returns an MxNx3 array
    # output_size = [input_img.shape[1], input_img.shape[0]]
    # output_path = "textureSyn/1/"
    output_path = "results/texture_synthesis/%d"%(args.case)
    patch_size = 40  # size of the patch (without the overlap)
    overlap_size = 10  # the width of the overlap region
    output_size = [300, 300]
    pbts = patchBasedTextureSynthesis(img_path, output_path, output_size, patch_size, overlap_size, in_windowStep=5,
                                      in_mirror_hor=True, in_mirror_vert=True, in_shapshots=False)
    target_img = pbts.resolveAll()
    return np.array(target_img)


def render(canvas_width, canvas_height, shapes, shape_groups, samples=2):
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width, # width
                 canvas_height, # height
                 samples,   # num_samples_x
                 samples,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    return img

def big_bounding_box(paths_n_stuff):
    """Finds a BB containing a collection of paths, Bezier path segments, and
    points (given as complex numbers)."""
    bbs = []
    for thing in paths_n_stuff:
        if is_path_segment(thing) or isinstance(thing, Path):
            bbs.append(thing.bbox())
        elif isinstance(thing, complex):
            bbs.append((thing.real, thing.real, thing.imag, thing.imag))
        else:
            try:
                complexthing = complex(thing)
                bbs.append((complexthing.real, complexthing.real,
                            complexthing.imag, complexthing.imag))
            except ValueError:
                raise TypeError(
                    "paths_n_stuff can only contains Path, CubicBezier, "
                    "QuadraticBezier, Line, and complex objects.")
    xmins, xmaxs, ymins, ymaxs = list(zip(*bbs))
    xmin = min(xmins)
    xmax = max(xmaxs)
    ymin = min(ymins)
    ymax = max(ymaxs)
    return xmin, xmax, ymin, ymax


def main(args):
    ## set device -> use cpu now since I haven't solved the nvcc issue
    pydiffvg.set_use_gpu(False)
    # pydiffvg.set_device(torch.device('cuda:1'))
    ## use L2 for now
    # perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())

    ## generate a texture synthesized
    target_img = texture_syn(args.target)
    tar_h, tar_w = target_img.shape[1], target_img.shape[0]
    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(args.svg_path)


    ## svgpathtools for checking the bounding box
    # paths, _, _ = svg2paths2(args.svg_path)
    # print(len(paths))
    # xmin, xmax, ymin, ymax = big_bounding_box(paths)
    # print(xmin, xmax, ymin, ymax)
    # input("check")


    print('tar h : %d tar w : %d'%(tar_h, tar_w))
    print('canvas h : %d canvas w : %d' % (canvas_height, canvas_width))
    scale_ratio = tar_h / canvas_height
    print("scale ratio : ", scale_ratio)
    # input("check")
    for path in shapes:
        path.points[..., 0] = path.points[..., 0] * scale_ratio
        path.points[..., 1] = path.points[..., 1] * scale_ratio

    init_img = render(tar_w, tar_h, shapes, shape_groups)
    pydiffvg.imwrite(init_img.cpu(), 'results/texture_synthesis/%d/init.png'%(args.case), gamma=2.2)
    # input("check")
    random.seed(1234)
    torch.manual_seed(1234)

    points_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    color_vars = []
    for group in shape_groups:
        group.fill_color.requires_grad = True
        color_vars.append(group.fill_color)
    # Optimize
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    target = torch.from_numpy(target_img).to(torch.float32) / 255.0
    target = target.pow(2.2)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2) # NHWC -> NCHW
    canvas_width, canvas_height = target.shape[3], target.shape[2]
    # print('canvas h : %d canvas w : %d' % (canvas_height, canvas_width))
    # input("check")

    for t in range(args.max_iter):
        print('iteration:', t)
        points_optim.zero_grad()
        color_optim.zero_grad()
        cur_img = render(canvas_width, canvas_height, shapes, shape_groups)
        pydiffvg.imwrite(cur_img.cpu(), 'results/texture_synthesis/%d/iter_%d.png'%(args.case, t), gamma=2.2)
        cur_img = cur_img[:, :, :3]
        cur_img = cur_img.unsqueeze(0)
        cur_img = cur_img.permute(0, 3, 1, 2) # NHWC -> NCHW

        ## perceptual loss
        # loss = perception_loss(cur_img, target)
        ## l2 loss
        loss = (cur_img - target).pow(2).mean()
        print('render loss:', loss.item())
        loss.backward()

        points_optim.step()
        color_optim.step()

        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)
        ## write svg
        if t % 10 == 0 or t == args.max_iter - 1:
            pydiffvg.save_svg('results/texture_synthesis/%d/iter_%d.svg'%(args.case, t),
                              canvas_width, canvas_height, shapes, shape_groups)

    ## render final result
    final_img = render(tar_h, tar_w, shapes, shape_groups)
    pydiffvg.imwrite(final_img.cpu(), 'results/texture_synthesis/%d/final.png'%(args.case), gamma=2.2)


    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        "results/texture_synthesis/%d/iter_%d.png"%(args.case), "-vb", "20M",
        "results/texture_synthesis/%d/out.mp4"%(args.case)])
    ## make gif
    make_gif("results/texture_synthesis/%d"%(args.case), "results/texture_synthesis/%d/out.gif"%(args.case), frame_every_X_steps=1, repeat_ending=3, total_iter=args.max_iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## target image path
    parser.add_argument("target", help="target image path")
    parser.add_argument("--svg-path", type=str, help="the corresponding svg file path")
    parser.add_argument("--max-iter", type=int, default=500, help="the max optimization iterations")
    parser.add_argument("--case", type=int, default=1, help="just the case id for a separate result folder")
    args = parser.parse_args()
    main(args)