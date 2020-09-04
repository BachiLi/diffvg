"""
"""
import os
import pydiffvg
import torch as th
import scipy.ndimage.filters as F


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


def main():
    pydiffvg.set_device(th.device('cuda:1'))

    # Load SVG
    svg = os.path.join("imgs", "peppers.svg")
    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(svg)

    # Save initial state
    ref = render(canvas_width, canvas_height, shapes, shape_groups)
    pydiffvg.imwrite(ref.cpu(), 'results/gaussian_blur/init.png', gamma=2.2)

    target = F.gaussian_filter(ref.cpu().numpy(), [10, 10, 0])
    target = th.from_numpy(target).to(ref.device)
    pydiffvg.imwrite(target.cpu(), 'results/gaussian_blur/target.png', gamma=2.2)

    # Collect variables to optimize
    points_vars = []
    width_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
        path.stroke_width.requires_grad = True
        width_vars.append(path.stroke_width)
    color_vars = []
    for group in shape_groups:
        # do not optimize alpha
        group.fill_color[..., :3].requires_grad = True
        color_vars.append(group.fill_color)

    # Optimize
    points_optim = th.optim.Adam(points_vars, lr=1.0)
    width_optim = th.optim.Adam(width_vars, lr=1.0)
    color_optim = th.optim.Adam(color_vars, lr=0.01)

    for t in range(20):
        print('\niteration:', t)
        points_optim.zero_grad()
        width_optim.zero_grad()
        color_optim.zero_grad()
        # Forward pass: render the image.
        img = render(canvas_width, canvas_height, shapes, shape_groups)
        # Save the intermediate render.
        pydiffvg.imwrite(img.cpu(), 'results/gaussian_blur/iter_{}.png'.format(t), gamma=2.2)
        loss = (img - target)[..., :3].pow(2).mean()

        print('alpha:', img[..., 3].mean().item())
        print('render loss:', loss.item())
    
        # Backpropagate the gradients.
        loss.backward()
    
        # Take a gradient descent step.
        points_optim.step()
        width_optim.step()
        color_optim.step()
        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)

    # Final render
    img = render(canvas_width, canvas_height, shapes, shape_groups)
    pydiffvg.imwrite(img.cpu(), 'results/gaussian_blur/final.png', gamma=2.2)

    # Convert the intermediate renderings to a video.
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        "results/gaussian_blur/iter_%d.png", "-vb", "20M",
        "results/gaussian_blur/out.mp4"])

if __name__ == "__main__":
    main()
