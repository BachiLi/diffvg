import pydiffvg
import torch
import skimage
import numpy as np

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 256, 256
circle = pydiffvg.Circle(radius = torch.tensor(40.0),
                         center = torch.tensor([128.0, 128.0]),
                         stroke_width = torch.tensor(5.0))
shapes = [circle]
circle_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
    fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]),
    stroke_color = torch.tensor([0.6, 0.3, 0.6, 0.8]))
shape_groups = [circle_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)

render = pydiffvg.RenderFunction.apply
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None,
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img.cpu(), 'results/single_circle_outline/target.png', gamma=2.2)
target = img.clone()

# Move the circle to produce initial guess
# normalize radius & center for easier learning rate
radius_n = torch.tensor(20.0 / 256.0, requires_grad=True)
center_n = torch.tensor([108.0 / 256.0, 138.0 / 256.0], requires_grad=True)
fill_color = torch.tensor([0.3, 0.2, 0.8, 1.0], requires_grad=True)
stroke_color = torch.tensor([0.4, 0.7, 0.5, 0.5], requires_grad=True)
stroke_width_n = torch.tensor(10.0 / 100.0, requires_grad=True)
circle.radius = radius_n * 256
circle.center = center_n * 256
circle.stroke_width = stroke_width_n * 100
circle_group.fill_color = fill_color
circle_group.stroke_color = stroke_color
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None,
             *scene_args)
pydiffvg.imwrite(img.cpu(), 'results/single_circle_outline/init.png', gamma=2.2)

# Optimize for radius & center
optimizer = torch.optim.Adam([radius_n, center_n, fill_color, stroke_color, stroke_width_n], lr=1e-2)
# Run 200 Adam iterations.
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    circle.radius = radius_n * 256
    circle.center = center_n * 256
    circle.stroke_width = stroke_width_n * 100
    circle_group.fill_color = fill_color
    circle_group.stroke_color = stroke_color
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(256,   # width
                 256,   # height
                 2,     # num_samples_x
                 2,     # num_samples_y
                 t+1,   # seed
                 None,
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), 'results/single_circle_outline/iter_{}.png'.format(t), gamma=2.2)
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    print('radius.grad:', radius_n.grad)
    print('center.grad:', center_n.grad)
    print('fill_color.grad:', fill_color.grad)
    print('stroke_color.grad:', stroke_color.grad)
    print('stroke_width.grad:', stroke_width_n.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current params.
    print('radius:', circle.radius)
    print('center:', circle.center)
    print('stroke_width:', circle.stroke_width)
    print('fill_color:', circle_group.fill_color)
    print('stroke_color:', circle_group.stroke_color)

# Render the final result.
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256,   # width
             256,   # height
             2,     # num_samples_x
             2,     # num_samples_y
             202,    # seed
             None,
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), 'results/single_circle_outline/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_circle_outline/iter_%d.png", "-vb", "20M",
    "results/single_circle_outline/out.mp4"])
