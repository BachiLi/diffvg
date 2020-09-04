"""
Simple utility to render an .svg to a .png
"""
import os
import argparse
import pydiffvg
import torch as th


def render(canvas_width, canvas_height, shapes, shape_groups):
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    return img


def main(args):
    pydiffvg.set_device(th.device('cuda:1'))

    # Load SVG
    svg = os.path.join(args.svg)
    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(svg)

    # Save initial state
    ref = render(canvas_width, canvas_height, shapes, shape_groups)
    pydiffvg.imwrite(ref.cpu(), args.out, gamma=2.2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", help="source SVG path")
    parser.add_argument("out", help="output image path")
    args = parser.parse_args()
    main(args)
