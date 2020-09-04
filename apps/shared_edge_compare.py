import pydiffvg
import diffvg
from matplotlib import cm
import matplotlib.pyplot as plt
import argparse
import torch

def normalize(x, min_, max_):
    range = max(abs(min_), abs(max_))
    return (x + range) / (2 * range)

def main(args):
    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(args.svg_file)

    w = int(canvas_width * args.size_scale)
    h = int(canvas_height * args.size_scale)

    pfilter = pydiffvg.PixelFilter(type = diffvg.FilterType.box,
                                   radius = torch.tensor(0.5))

    use_prefiltering = False
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        filter = pfilter,
        use_prefiltering = use_prefiltering)

    num_samples_x = 16
    num_samples_y = 16
    render = pydiffvg.RenderFunction.apply
    img = render(w, # width
                 h, # height
                 num_samples_x, # num_samples_x
                 num_samples_y, # num_samples_y
                 0, # seed
                 None,
                 *scene_args)
    pydiffvg.imwrite(img.cpu(), 'results/finite_difference_comp/img.png', gamma=1.0)

    epsilon = 0.1
    def perturb_scene(axis, epsilon):
        shapes[2].points[:, axis] += epsilon
        # for s in shapes:
        #     if isinstance(s, pydiffvg.Circle):
        #         s.center[axis] += epsilon
        #     elif isinstance(s, pydiffvg.Ellipse):
        #         s.center[axis] += epsilon
        #     elif isinstance(s, pydiffvg.Path):
        #         s.points[:, axis] += epsilon
        #     elif isinstance(s, pydiffvg.Polygon):
        #         s.points[:, axis] += epsilon
        #     elif isinstance(s, pydiffvg.Rect):
        #         s.p_min[axis] += epsilon
        #         s.p_max[axis] += epsilon
        # for s in shape_groups:
        #     if isinstance(s.fill_color, pydiffvg.LinearGradient):
        #         s.fill_color.begin[axis] += epsilon
        #         s.fill_color.end[axis] += epsilon

    perturb_scene(0, epsilon)
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        filter = pfilter,
        use_prefiltering = use_prefiltering)
    render = pydiffvg.RenderFunction.apply
    img0 = render(w, # width
                  h, # height
                  num_samples_x,   # num_samples_x
                  num_samples_y,   # num_samples_y
                  0,   # seed
                  None,
                  *scene_args)

    forward_diff = (img0 - img) / (epsilon)
    forward_diff = forward_diff.sum(axis = 2)
    x_diff_max = 1.5
    x_diff_min = -1.5
    print(forward_diff.max())
    print(forward_diff.min())
    forward_diff = cm.viridis(normalize(forward_diff, x_diff_min, x_diff_max).cpu().numpy())
    pydiffvg.imwrite(forward_diff, 'results/finite_difference_comp/shared_edge_forward_diff.png', gamma=1.0)

    perturb_scene(0, -2 * epsilon)
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        filter = pfilter,
        use_prefiltering = use_prefiltering)
    img1 = render(w, # width
                  h, # height
                  num_samples_x,   # num_samples_x
                  num_samples_y,   # num_samples_y
                  0,   # seed
                  None,
                  *scene_args)
    backward_diff = (img - img1) / (epsilon)
    backward_diff = backward_diff.sum(axis = 2)
    print(backward_diff.max())
    print(backward_diff.min())
    backward_diff = cm.viridis(normalize(backward_diff, x_diff_min, x_diff_max).cpu().numpy())
    pydiffvg.imwrite(backward_diff, 'results/finite_difference_comp/shared_edge_backward_diff.png', gamma=1.0)
    perturb_scene(0, epsilon)

    num_samples_x = 4
    num_samples_y = 4
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        filter = pfilter,
        use_prefiltering = use_prefiltering)
    render_grad = pydiffvg.RenderFunction.render_grad
    img_grad = render_grad(torch.ones(h, w, 4),
                           w, # width
                           h, # height
                           num_samples_x, # num_samples_x
                           num_samples_y, # num_samples_y
                           0, # seed
                           *scene_args)
    print(img_grad[:, :, 0].max())
    print(img_grad[:, :, 0].min())
    x_diff = cm.viridis(normalize(img_grad[:, :, 0], x_diff_min, x_diff_max).cpu().numpy())
    pydiffvg.imwrite(x_diff, 'results/finite_difference_comp/ours_x_diff.png', gamma=1.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("svg_file", help="source SVG path")
    parser.add_argument("--size_scale", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
