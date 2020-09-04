"""Retargets an .svg using image-domain seam carving to shrink it."""
import os
import pydiffvg
import argparse
import torch as th
import scipy.ndimage.filters as filters
import numba
import numpy as np
import skimage.io


def energy(im):
    """Compute image energy.

    Args:
        im(np.ndarray) with shape [h, w, 3]: input image.

    Returns:
        (np.ndarray) with shape [h, w]: energy map.
    """
    f_dx = np.array([
        [-1, 0, 1 ],
        [-2, 0, 2 ],
        [-1, 0, 1 ],
    ])
    f_dy = f_dx.T
    dx = filters.convolve(im.mean(2), f_dx)
    dy = filters.convolve(im.mean(2), f_dy)

    return np.abs(dx) + np.abs(dy)


@numba.jit(nopython=True)
def min_seam(e):
    """Finds the seam with minimal cost in an energy map.

    Args:
        e(np.ndarray) with shape [h, w]: energy map.
    
    Returns:
        min_e(np.ndarray) with shape [h, w]: for all (y,x) min_e[y, x]
            is the cost of the minimal seam from 0 to y (top to bottom).
            The minimal seam can be found by looking at the last row of min_e.
            This is computed by dynamic programming.
        argmin_e(np.ndarray) with shape [h, w]: for all (y,x) argmin_e[y, x]
            contains the x coordinate corresponding to this seam in the
            previous row (y-1). We use this for backtracking.
    """
    # initialize to local energy
    min_e = e.copy()
    argmin_e = np.zeros_like(e, dtype=np.int64)

    h, w =  e.shape

    # propagate vertically
    for y in range(1, h):
        for x in range(w):
            if x == 0:
                idx = np.argmin(e[y-1, x:x+2])
                argmin_e[y, x] = idx + x
                mini = e[y-1, x + idx]
            elif x == w-1:
                idx = np.argmin(e[y-1, x-1:x+1])
                argmin_e[y, x] = idx + x - 1
                mini = e[y-1, x + idx - 1]
            else:
                idx = np.argmin(e[y-1, x-1:x+2])
                argmin_e[y, x] = idx + x - 1
                mini = e[y-1, x + idx - 1]

            min_e[y, x] = min_e[y, x] + mini

    return min_e, argmin_e


def carve_seam(im):
    """Carves a vertical seam in an image, reducing it's horizontal size by 1.

    Args:
        im(np.ndarray) with shape [h, w, 3]: input image.

    Returns:
        (np.ndarray) with shape [h, w-1, 1]: the image with one seam removed.
    """

    e = energy(im)
    min_e, argmin_e = min_seam(e)
    h, w =  im.shape[:2]

    # boolean flags for the pixels to preserve
    to_keep = np.ones((h, w), dtype=np.bool)

    # get lowest energy (from last row)
    x = np.argmin(min_e[-1])
    print("carving seam", x, "with energy", min_e[-1, x])

    # backtract to identify the seam
    for y in range(h-1, -1, -1):
        # remove seam pixel
        to_keep[y, x] = False
        x = argmin_e[y, x]

    # replicate mask over color channels
    to_keep = np.stack(3*[to_keep], axis=2)
    new_im = im[to_keep].reshape((h, w-1, 3))
    return new_im


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


def vector_rescale(shapes, scale_x=1.00, scale_y=1.00):
    new_shapes = []
    for path in shapes:
        path.points[..., 0] *= scale_x
        path.points[..., 1] *= scale_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", default=os.path.join("imgs", "hokusai.svg"))
    parser.add_argument("--optim_steps", default=10, type=int)
    parser.add_argument("--lr", default=1e-1, type=int)
    args = parser.parse_args()

    name = os.path.splitext(os.path.basename(args.svg))[0]
    root = os.path.join("results", "seam_carving", name)
    svg_root = os.path.join(root, "svg")
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "svg"), exist_ok=True)

    pydiffvg.set_use_gpu(False)
    # pydiffvg.set_device(th.device('cuda'))

    # Load SVG
    print("loading svg %s" % args.svg)
    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(args.svg)
    print("done loading")

    max_size = 512
    scale_factor = max_size / max(canvas_width, canvas_height)
    print("rescaling from %dx%d with scale %f" % (canvas_width, canvas_height, scale_factor))
    canvas_width = int(canvas_width*scale_factor)
    canvas_height = int(canvas_height*scale_factor)
    print("new shape %dx%d" % (canvas_width, canvas_height))
    vector_rescale(shapes, scale_x=scale_factor, scale_y=scale_factor)

    # Shrink image by 33 %
    # num_seams_to_remove = 2
    num_seams_to_remove = canvas_width // 3
    new_canvas_width  = canvas_width - num_seams_to_remove
    scaling =  new_canvas_width * 1.0 / canvas_width

    # Naive scaling baseline
    print("rendering naive rescaling...")
    vector_rescale(shapes, scale_x=scaling)
    resized = render(new_canvas_width, canvas_height, shapes, shape_groups)
    pydiffvg.imwrite(resized.cpu(), os.path.join(root, 'uniform_scaling.png'), gamma=2.2)
    pydiffvg.save_svg(os.path.join(svg_root, 'uniform_scaling.svg') , canvas_width,
                      canvas_height, shapes, shape_groups, use_gamma=False)
    vector_rescale(shapes, scale_x=1.0/scaling)  # bring back original coordinates
    print("saved naiving scaling")

    # Save initial state
    print("rendering initial state...")
    im = render(canvas_width, canvas_height, shapes, shape_groups)
    pydiffvg.imwrite(im.cpu(), os.path.join(root, 'init.png'), gamma=2.2)
    pydiffvg.save_svg(os.path.join(svg_root, 'init.svg'), canvas_width,
                      canvas_height, shapes, shape_groups, use_gamma=False)
    print("saved initial state")

    # Optimize
    # color_optim = th.optim.Adam(color_vars, lr=0.01)

    retargeted = im[..., :3].cpu().numpy()
    previous_width = canvas_width
    print("carving seams")
    for seam_idx in range(num_seams_to_remove):
        print('\nseam', seam_idx+1, 'of', num_seams_to_remove)

        # Remove a seam
        retargeted = carve_seam(retargeted)

        current_width = canvas_width - seam_idx - 1
        scale_factor = current_width * 1.0 / previous_width
        previous_width = current_width

        padded = np.zeros((canvas_height, canvas_width, 4))
        padded[:, :-seam_idx-1, :3] = retargeted
        padded[:, :-seam_idx-1, -1] = 1.0  # alpha
        padded = th.from_numpy(padded).to(im.device)

        # Remap points to the smaller canvas and
        # collect variables to optimize
        points_vars = []
        # width_vars = []
        mini, maxi = canvas_width, 0
        for path in shapes:
            path.points.requires_grad = False
            x = path.points[..., 0]
            y = path.points[..., 1]
            # rescale

            x = x * scale_factor

            # clip to canvas
            path.points[..., 0] = th.clamp(x, 0, current_width)
            path.points[..., 1] = th.clamp(y, 0, canvas_height)

            path.points.requires_grad = True
            points_vars.append(path.points)
            path.stroke_width.requires_grad = True
            # width_vars.append(path.stroke_width)

            mini = min(mini, path.points.min().item())
            maxi = max(maxi, path.points.max().item())
        print("points", mini, maxi, "scale", scale_factor)

        # recreate an optimizer so we don't carry over the previous update
        # (momentum)?
        geom_optim = th.optim.Adam(points_vars, lr=args.lr)

        for step in range(args.optim_steps):
            geom_optim.zero_grad()

            img = render(canvas_width, canvas_height, shapes, shape_groups,
                         samples=2)

            pydiffvg.imwrite(
                img.cpu(), 
                os.path.join(root, "seam_%03d_iter_%02d.png" % (seam_idx, step)), gamma=2.2)

            # NO alpha
            loss = (img - padded)[..., :3].pow(2).mean()
            # loss = (img - padded).pow(2).mean()
            print('render loss:', loss.item())

            # Backpropagate the gradients.
            loss.backward()

            # Take a gradient descent step.
            geom_optim.step()
        pydiffvg.save_svg(os.path.join(svg_root, "seam%03d.svg" % seam_idx),
                          canvas_width-seam_idx, canvas_height, shapes,
                          shape_groups, use_gamma=False)

        for path in shapes:
            mini = min(mini, path.points.min().item())
            maxi = max(maxi, path.points.max().item())
        print("points", mini, maxi)

    img = render(canvas_width, canvas_height, shapes, shape_groups)
    img = img[:, :-num_seams_to_remove]

    pydiffvg.imwrite(img.cpu(), os.path.join(root, 'final.png'),
                     gamma=2.2)
    pydiffvg.imwrite(retargeted, os.path.join(root, 'ref.png'),
                     gamma=2.2)

    pydiffvg.save_svg(os.path.join(svg_root, 'final.svg'),
                      canvas_width-seam_idx, canvas_height, shapes,
                      shape_groups, use_gamma=False)

    # Convert the intermediate renderings to a video.
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i", os.path.join(root, "seam_%03d_iter_00.png"), "-vb", "20M",
         os.path.join(root, "out.mp4")])


if __name__ == "__main__":
    main()
