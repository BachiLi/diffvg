# python finite_difference_comp.py imgs/tiger.svg 
# python finite_difference_comp.py --use_prefiltering True imgs/tiger.svg 
# python finite_difference_comp.py imgs/boston.svg
# python finite_difference_comp.py --use_prefiltering True imgs/boston.svg
# python finite_difference_comp.py imgs/contour.svg
# python finite_difference_comp.py --use_prefiltering True imgs/contour.svg
# python finite_difference_comp.py --size_scale 0.5 --clamping_factor 0.05 imgs/hawaii.svg
# python finite_difference_comp.py --size_scale 0.5 --clamping_factor 0.05 --use_prefiltering True imgs/hawaii.svg
# python finite_difference_comp.py imgs/mcseem2.svg
# python finite_difference_comp.py --use_prefiltering True imgs/mcseem2.svg
# python finite_difference_comp.py imgs/reschart.svg
# python finite_difference_comp.py --use_prefiltering True imgs/reschart.svg

import pydiffvg
import diffvg
from matplotlib import cm
import matplotlib.pyplot as plt
import argparse
import torch

pydiffvg.set_print_timing(True)
#pydiffvg.set_use_gpu(False)

def normalize(x, min_, max_):
    range = max(abs(min_), abs(max_))
    return (x + range) / (2 * range)

def main(args):
    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(args.svg_file)

    w = int(canvas_width * args.size_scale)
    h = int(canvas_height * args.size_scale)

    print(w, h)
    curve_counts = 0
    for s in shapes:
        if isinstance(s, pydiffvg.Circle):
            curve_counts += 1
        elif isinstance(s, pydiffvg.Ellipse):
            curve_counts += 1
        elif isinstance(s, pydiffvg.Path):
            curve_counts += len(s.num_control_points)
        elif isinstance(s, pydiffvg.Polygon):
            curve_counts += len(s.points) - 1
            if s.is_closed:
                curve_counts += 1
        elif isinstance(s, pydiffvg.Rect):
            curve_counts += 1
    print('curve_counts:', curve_counts)

    pfilter = pydiffvg.PixelFilter(type = diffvg.FilterType.box,
                                   radius = torch.tensor(0.5))

    use_prefiltering = args.use_prefiltering
    print('use_prefiltering:', use_prefiltering)

    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        filter = pfilter,
        use_prefiltering = use_prefiltering)

    num_samples_x = args.num_spp
    num_samples_y = args.num_spp
    if (use_prefiltering):
        num_samples_x = 1
        num_samples_y = 1

    render = pydiffvg.RenderFunction.apply
    img = render(w, # width
                 h, # height
                 num_samples_x, # num_samples_x
                 num_samples_y, # num_samples_y
                 0, # seed
                 None, # background_image
                 *scene_args)
    pydiffvg.imwrite(img.cpu(), 'results/finite_difference_comp/img.png', gamma=1.0)

    epsilon = 0.1
    def perturb_scene(axis, epsilon):
        for s in shapes:
            if isinstance(s, pydiffvg.Circle):
                s.center[axis] += epsilon
            elif isinstance(s, pydiffvg.Ellipse):
                s.center[axis] += epsilon
            elif isinstance(s, pydiffvg.Path):
                s.points[:, axis] += epsilon
            elif isinstance(s, pydiffvg.Polygon):
                s.points[:, axis] += epsilon
            elif isinstance(s, pydiffvg.Rect):
                s.p_min[axis] += epsilon
                s.p_max[axis] += epsilon
        for s in shape_groups:
            if isinstance(s.fill_color, pydiffvg.LinearGradient):
                s.fill_color.begin[axis] += epsilon
                s.fill_color.end[axis] += epsilon

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
                  None, # background_image
                  *scene_args)

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
                  None, # background_image
                  *scene_args)
    x_diff = (img0 - img1) / (2 * epsilon)
    x_diff = x_diff.sum(axis = 2)
    x_diff_max = x_diff.max() * args.clamping_factor
    x_diff_min = x_diff.min() * args.clamping_factor
    print(x_diff.max())
    print(x_diff.min())
    x_diff = cm.viridis(normalize(x_diff, x_diff_min, x_diff_max).cpu().numpy())
    pydiffvg.imwrite(x_diff, 'results/finite_difference_comp/finite_x_diff.png', gamma=1.0)

    perturb_scene(0, epsilon)

    perturb_scene(1, epsilon)
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
                  None, # background_image
                  *scene_args)

    perturb_scene(1, -2 * epsilon)
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        filter = pfilter,
        use_prefiltering = use_prefiltering)
    img1 = render(w, # width
                  h, # height
                  num_samples_x,   # num_samples_x
                  num_samples_y,   # num_samples_y
                  0,   # seed
                  None, # background_image
                  *scene_args)
    y_diff = (img0 - img1) / (2 * epsilon)
    y_diff = y_diff.sum(axis = 2)
    y_diff_max = y_diff.max() * args.clamping_factor
    y_diff_min = y_diff.min() * args.clamping_factor
    y_diff = cm.viridis(normalize(y_diff, y_diff_min, y_diff_max).cpu().numpy())
    pydiffvg.imwrite(y_diff, 'results/finite_difference_comp/finite_y_diff.png', gamma=1.0)
    perturb_scene(1, epsilon)

    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        filter = pfilter,
        use_prefiltering = use_prefiltering)
    render_grad = pydiffvg.RenderFunction.render_grad
    img_grad = render_grad(torch.ones(h, w, 4, device = pydiffvg.get_device()),
                           w, # width
                           h, # height
                           num_samples_x, # num_samples_x
                           num_samples_y, # num_samples_y
                           0, # seed
                           None, # background_image
                           *scene_args)
    print(img_grad[:, :, 0].max())
    print(img_grad[:, :, 0].min())
    x_diff = cm.viridis(normalize(img_grad[:, :, 0], x_diff_min, x_diff_max).cpu().numpy())
    y_diff = cm.viridis(normalize(img_grad[:, :, 1], y_diff_min, y_diff_max).cpu().numpy())
    pydiffvg.imwrite(x_diff, 'results/finite_difference_comp/ours_x_diff.png', gamma=1.0)
    pydiffvg.imwrite(y_diff, 'results/finite_difference_comp/ours_y_diff.png', gamma=1.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("svg_file", help="source SVG path")
    parser.add_argument("--size_scale", type=float, default=1.0)
    parser.add_argument("--clamping_factor", type=float, default=0.1)
    parser.add_argument("--num_spp", type=int, default=4)
    parser.add_argument("--use_prefiltering", type=bool, default=False)
    args = parser.parse_args()
    main(args)
