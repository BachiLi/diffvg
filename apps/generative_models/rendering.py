import os
import torch as th
import torch.multiprocessing as mp
import threading as mt
import numpy as np
import random

import ttools

import pydiffvg
import time


def render(canvas_width, canvas_height, shapes, shape_groups, samples=2,
           seed=None):
    if seed is None:
        seed = random.randint(0, 1000000)
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width, canvas_height, samples, samples,
                  seed,   # seed
                  None,  # background image
                  *scene_args)
    return img


def opacityStroke2diffvg(strokes, canvas_size=128, debug=False, relative=True,
                         force_cpu=True):

    dev = strokes.device
    if force_cpu:
        strokes = strokes.to("cpu")


    # pydiffvg.set_use_gpu(False)
    # if strokes.is_cuda:
    #     pydiffvg.set_use_gpu(True)

    """Rasterize strokes given in (dx, dy, opacity) sequence format."""
    bs, nsegs, dims = strokes.shape
    out = []

    start = time.time()
    for batch_idx, stroke in enumerate(strokes):

        if relative:  # Absolute coordinates
            all_points = stroke[..., :2].cumsum(0)
        else:
            all_points = stroke[..., :2]

        all_opacities = stroke[..., 2]

        # Transform from [-1, 1] to canvas coordinates
        # Make sure points are in canvas
        all_points = 0.5*(all_points + 1.0) * canvas_size
        # all_points = th.clamp(0.5*(all_points + 1.0), 0, 1) * canvas_size

        # Avoid overlapping points
        eps = 1e-4
        all_points = all_points + eps*th.randn_like(all_points)

        shapes = []
        shape_groups = []

        for start_idx in range(0, nsegs-1):
            points = all_points[start_idx:start_idx+2].contiguous().float()
            opacity = all_opacities[start_idx]

            num_ctrl_pts = th.zeros(points.shape[0] - 1, dtype=th.int32)
            width = th.ones(1)

            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts, points=points,
                stroke_width=width, is_closed=False)

            shapes.append(path)

            color = th.cat([th.ones(3, device=opacity.device),
                            opacity.unsqueeze(0)], 0)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=th.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=color)
            shape_groups.append(path_group)

        # Rasterize only if there are shapes
        if shapes:
            inner_start = time.time()
            out.append(render(canvas_size, canvas_size, shapes, shape_groups,
                              samples=4))
            if debug:
                inner_elapsed = time.time() - inner_start
                print("diffvg call took %.2fms" % inner_elapsed)
        else:
            out.append(th.zeros(canvas_size, canvas_size, 4,
                                device=strokes.device))

    if debug:
        elapsed = (time.time() - start)*1000
        print("rendering took %.2fms" % elapsed)
    images = th.stack(out, 0).permute(0, 3, 1, 2).contiguous()

    # Return data on the same device as input
    return images.to(dev)


def stroke2diffvg(strokes, canvas_size=128):
    """Rasterize strokes given some sequential data."""
    bs, nsegs, dims = strokes.shape
    out = []
    for stroke_idx, stroke in enumerate(strokes):
        end_of_stroke = stroke[:, 4] == 1
        last = end_of_stroke.cpu().numpy().argmax()
        stroke = stroke[:last+1, :]
        # stroke = stroke[~end_of_stroke]
        # TODO: stop at the first end of stroke
        # import ipdb; ipdb.set_trace()
        split_idx = stroke[:, 3].nonzero().squeeze(1)

        # Absolute coordinates
        all_points = stroke[..., :2].cumsum(0)

        # Transform to canvas coordinates
        all_points[..., 0] += 0.5
        all_points[..., 0] *= canvas_size
        all_points[..., 1] += 0.5
        all_points[..., 1] *= canvas_size

        # Make sure points are in canvas
        all_points[..., :2] = th.clamp(all_points[..., :2], 0, canvas_size)

        shape_groups = []
        shapes = []
        start_idx = 0

        for count, end_idx in enumerate(split_idx):
            points = all_points[start_idx:end_idx+1].contiguous().float()

            if points.shape[0] <= 2:  # we need at least 2 points for a line
                continue

            num_ctrl_pts = th.zeros(points.shape[0] - 1, dtype=th.int32)
            width = th.ones(1)
            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts, points=points,
                stroke_width=width, is_closed=False)

            start_idx = end_idx+1
            shapes.append(path)

            color = th.ones(4, 1)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=th.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=color)
            shape_groups.append(path_group)

        # Rasterize
        if shapes:
            # draw only if there are shapes
            out.append(render(canvas_size, canvas_size, shapes, shape_groups, samples=2))
        else:
            out.append(th.zeros(canvas_size, canvas_size, 4,
                                device=strokes.device))

    return th.stack(out, 0).permute(0, 3, 1, 2)[:, :3].contiguous()


def line_render(all_points, all_widths, all_alphas, force_cpu=True,
                canvas_size=32, colors=None):
    dev = all_points.device
    if force_cpu:
        all_points = all_points.to("cpu")
        all_widths = all_widths.to("cpu")
        all_alphas = all_alphas.to("cpu")

        if colors is not None:
            colors = colors.to("cpu")

    all_points = 0.5*(all_points + 1.0) * canvas_size

    eps = 1e-4
    all_points = all_points + eps*th.randn_like(all_points)

    bs, num_segments, _, _ = all_points.shape
    n_out = 3 if colors is not None else 1
    output = th.zeros(bs, n_out, canvas_size, canvas_size,
                      device=all_points.device)

    scenes = []
    for k in range(bs):
        shapes = []
        shape_groups = []
        for p in range(num_segments):
            points = all_points[k, p].contiguous().cpu()
            num_ctrl_pts = th.zeros(1, dtype=th.int32)
            width = all_widths[k, p].cpu()
            alpha = all_alphas[k, p].cpu()
            if colors is not None:
                color = colors[k, p]
            else:
                color = th.ones(3, device=alpha.device)

            color = th.cat([color, alpha.view(1,)])

            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts, points=points,
                stroke_width=width, is_closed=False)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=th.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=color)
            shape_groups.append(path_group)

        # Rasterize
        scenes.append((canvas_size, canvas_size, shapes, shape_groups))
        raster = render(canvas_size, canvas_size, shapes, shape_groups,
                        samples=2)
        raster = raster.permute(2, 0, 1).view(4, canvas_size, canvas_size)

        alpha = raster[3:4]
        if colors is not None:  # color output
            image = raster[:3]
            alpha = alpha.repeat(3, 1, 1)
        else:
            image = raster[:1]

        # alpha compositing
        image = image*alpha
        output[k] = image

    output = output.to(dev)

    return output, scenes


def bezier_render(all_points, all_widths, all_alphas, force_cpu=True,
                  canvas_size=32, colors=None):
    dev = all_points.device
    if force_cpu:
        all_points = all_points.to("cpu")
        all_widths = all_widths.to("cpu")
        all_alphas = all_alphas.to("cpu")

        if colors is not None:
            colors = colors.to("cpu")

    all_points = 0.5*(all_points + 1.0) * canvas_size

    eps = 1e-4
    all_points = all_points + eps*th.randn_like(all_points)

    bs, num_strokes, num_pts, _ = all_points.shape
    num_segments = (num_pts - 1) // 3
    n_out = 3 if colors is not None else 1
    output = th.zeros(bs, n_out, canvas_size, canvas_size,
                      device=all_points.device)

    scenes = []
    for k in range(bs):
        shapes = []
        shape_groups = []
        for p in range(num_strokes):
            points = all_points[k, p].contiguous().cpu()
            # bezier
            num_ctrl_pts = th.zeros(num_segments, dtype=th.int32) + 2
            width = all_widths[k, p].cpu()
            alpha = all_alphas[k, p].cpu()
            if colors is not None:
                color = colors[k, p]
            else:
                color = th.ones(3, device=alpha.device)

            color = th.cat([color, alpha.view(1,)])

            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts, points=points,
                stroke_width=width, is_closed=False)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=th.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=color)
            shape_groups.append(path_group)

        # Rasterize
        scenes.append((canvas_size, canvas_size, shapes, shape_groups))
        raster = render(canvas_size, canvas_size, shapes, shape_groups,
                        samples=2)
        raster = raster.permute(2, 0, 1).view(4, canvas_size, canvas_size)

        alpha = raster[3:4]
        if colors is not None:  # color output
            image = raster[:3]
            alpha = alpha.repeat(3, 1, 1)
        else:
            image = raster[:1]

        # alpha compositing
        image = image*alpha
        output[k] = image

    output = output.to(dev)

    return output, scenes
