import pydiffvg
import argparse
import ttools.modules
import torch
import skimage.io

gamma = 1.0

def main(args):
    perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())

    target = torch.from_numpy(skimage.io.imread(args.target)).to(torch.float32) / 255.0
    target = target.pow(gamma)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2) # NHWC -> NCHW

    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(args.svg)
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)

    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None, # bg
                 *scene_args)
    # The output image is in linear RGB space. Do Gamma correction before saving the image.
    pydiffvg.imwrite(img.cpu(), 'results/refine_svg/init.png', gamma=gamma)

    points_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    color_vars = {}
    for group in shape_groups:
        group.fill_color.requires_grad = True
        color_vars[group.fill_color.data_ptr()] = group.fill_color
    color_vars = list(color_vars.values())

    # Optimize
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    # Adam iterations.
    for t in range(args.num_iter):
        print('iteration:', t)
        points_optim.zero_grad()
        color_optim.zero_grad()
        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, # width
                     canvas_height, # height
                     2,   # num_samples_x
                     2,   # num_samples_y
                     0,   # seed
                     None, # bg
                     *scene_args)
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        # Save the intermediate render.
        pydiffvg.imwrite(img.cpu(), 'results/refine_svg/iter_{}.png'.format(t), gamma=gamma)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        if args.use_lpips_loss:
            loss = perception_loss(img, target)
        else:
            loss = (img - target).pow(2).mean()
        print('render loss:', loss.item())
    
        # Backpropagate the gradients.
        loss.backward()
    
        # Take a gradient descent step.
        points_optim.step()
        color_optim.step()
        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)

        if t % 10 == 0 or t == args.num_iter - 1:
            pydiffvg.save_svg('results/refine_svg/iter_{}.svg'.format(t),
                              canvas_width, canvas_height, shapes, shape_groups)

    # Render the final result.
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None, # bg
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), 'results/refine_svg/final.png'.format(t), gamma=gamma)
    # Convert the intermediate renderings to a video.
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        "results/refine_svg/iter_%d.png", "-vb", "20M",
        "results/refine_svg/out.mp4"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", help="source SVG path")
    parser.add_argument("target", help="target image path")
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true')
    parser.add_argument("--num_iter", type=int, default=250)
    args = parser.parse_args()
    main(args)
