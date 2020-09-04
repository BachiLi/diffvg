import diffvg
import pydiffvg
import torch
import skimage
import numpy as np

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width = 256
canvas_height = 256
circle = pydiffvg.Circle(radius = torch.tensor(40.0),
                         center = torch.tensor([128.0, 128.0]))
shapes = [circle]
circle_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
    fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))
shape_groups = [circle_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width=canvas_width,
    canvas_height=canvas_height,
    shapes=shapes,
    shape_groups=shape_groups,
    filter=pydiffvg.PixelFilter(type = diffvg.FilterType.hann,
                                radius = torch.tensor(8.0)))

render = pydiffvg.RenderFunction.apply
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None,
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img.cpu(), 'results/optimize_pixel_filter/target.png', gamma=2.2)
target = img.clone()

# Change the pixel filter radius
radius = torch.tensor(1.0, requires_grad = True)
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width=canvas_width,
    canvas_height=canvas_height,
    shapes=shapes,
    shape_groups=shape_groups,
    filter=pydiffvg.PixelFilter(type = diffvg.FilterType.hann,
                                radius = radius))
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None,
             *scene_args)
pydiffvg.imwrite(img.cpu(), 'results/optimize_pixel_filter/init.png', gamma=2.2)

# Optimize for radius & center
optimizer = torch.optim.Adam([radius], lr=1.0)
# Run 100 Adam iterations.
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        shapes=shapes,
        shape_groups=shape_groups,
        filter=pydiffvg.PixelFilter(type = diffvg.FilterType.hann,
                                    radius = radius))
    img = render(256,   # width
                 256,   # height
                 2,     # num_samples_x
                 2,     # num_samples_y
                 t+1,   # seed
                 None,
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), 'results/optimize_pixel_filter/iter_{}.png'.format(t), gamma=2.2)
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    print('radius.grad:', radius.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current params.
    print('radius:', radius)

# Render the final result.
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width=canvas_width,
    canvas_height=canvas_height,
    shapes=shapes,
    shape_groups=shape_groups,
    filter=pydiffvg.PixelFilter(type = diffvg.FilterType.hann,
                                radius = radius))
img = render(256,   # width
             256,   # height
             2,     # num_samples_x
             2,     # num_samples_y
             102,    # seed
             None,
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), 'results/optimize_pixel_filter/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/optimize_pixel_filter/iter_%d.png", "-vb", "20M",
    "results/optimize_pixel_filter/out.mp4"])
