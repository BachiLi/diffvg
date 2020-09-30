import pydiffvg
import torch
import skimage
import numpy as np

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 256, 256
color = pydiffvg.LinearGradient(\
    begin = torch.tensor([50.0, 50.0]),
    end = torch.tensor([200.0, 200.0]),
    offsets = torch.tensor([0.0, 1.0]),
    stop_colors = torch.tensor([[0.2, 0.5, 0.7, 1.0],
                                [0.7, 0.2, 0.5, 1.0]]))
circle = pydiffvg.Circle(radius = torch.tensor(40.0),
                         center = torch.tensor([128.0, 128.0]))
shapes = [circle]
circle_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]), fill_color = color)
shape_groups = [circle_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)

render = pydiffvg.RenderFunction.apply
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img.cpu(), 'results/single_gradient/target.png', gamma=2.2)
target = img.clone()

# Move the circle to produce initial guess
# normalize radius & center for easier learning rate
radius_n = torch.tensor(20.0 / 256.0, requires_grad=True)
center_n = torch.tensor([108.0 / 256.0, 138.0 / 256.0], requires_grad=True)
begin_n = torch.tensor([100.0 / 256.0, 100.0 / 256.0], requires_grad=True)
end_n = torch.tensor([150.0 / 256.0, 150.0 / 256.0], requires_grad=True)
stop_colors = torch.tensor([[0.1, 0.9, 0.2, 1.0],
                            [0.5, 0.3, 0.6, 1.0]], requires_grad=True)
color.begin = begin_n * 256
color.end = end_n * 256
color.stop_colors = stop_colors
circle.radius = radius_n * 256
circle.center = center_n * 256
circle_group.fill_color = color
shapes = [circle]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None, # background_image
             *scene_args)
pydiffvg.imwrite(img.cpu(), 'results/single_gradient/init.png', gamma=2.2)

# Optimize for radius & center
optimizer = torch.optim.Adam([radius_n, center_n, begin_n, end_n, stop_colors], lr=1e-2)
# Run 50 Adam iterations.
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    color.begin = begin_n * 256
    color.end = end_n * 256
    color.stop_colors = stop_colors
    circle.radius = radius_n * 256
    circle.center = center_n * 256
    circle_group.fill_color = color
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(256,   # width
                 256,   # height
                 2,     # num_samples_x
                 2,     # num_samples_y
                 t+1,   # seed
                 None, # background_image
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), 'results/single_gradient/iter_{}.png'.format(t), gamma=2.2)
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    print('radius.grad:', radius_n.grad)
    print('center.grad:', center_n.grad)
    print('begin.grad:', begin_n.grad)
    print('end.grad:', end_n.grad)
    print('stop_colors.grad:', stop_colors.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current params.
    print('radius:', circle.radius)
    print('center:', circle.center)
    print('begin:', begin_n)
    print('end:', end_n)
    print('stop_colors:', stop_colors)

# Render the final result.
color.begin = begin_n * 256
color.end = end_n * 256
color.stop_colors = stop_colors
circle.radius = radius_n * 256
circle.center = center_n * 256
circle_group.fill_color = color
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256,   # width
             256,   # height
             2,     # num_samples_x
             2,     # num_samples_y
             52,    # seed
             None, # background_image
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), 'results/single_gradient/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_gradient/iter_%d.png", "-vb", "20M",
    "results/single_gradient/out.mp4"])
