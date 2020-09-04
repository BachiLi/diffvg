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
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)

render = pydiffvg.RenderFunction.apply
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None,
             *scene_args)
img = img / 256 # Normalize SDF to [0, 1]
pydiffvg.imwrite(img.cpu(), 'results/single_circle_sdf/target.png')
target = img.clone()

# Move the circle to produce initial guess
# normalize radius & center for easier learning rate
radius_n = torch.tensor(20.0 / 256.0, requires_grad=True)
center_n = torch.tensor([108.0 / 256.0, 138.0 / 256.0], requires_grad=True)
color = torch.tensor([0.3, 0.2, 0.8, 1.0], requires_grad=True)
circle.radius = radius_n * 256
circle.center = center_n * 256
circle_group.fill_color = color
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None,
             *scene_args)
img = img / 256 # Normalize SDF to [0, 1]
pydiffvg.imwrite(img.cpu(), 'results/single_circle_sdf/init.png')

# Optimize for radius & center
optimizer = torch.optim.Adam([radius_n, center_n, color], lr=1e-2)
# Run 100 Adam iterations.
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    circle.radius = radius_n * 256
    circle.center = center_n * 256
    circle_group.fill_color = color
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        output_type = pydiffvg.OutputType.sdf)
    img = render(256,   # width
                 256,   # height
                 2,     # num_samples_x
                 2,     # num_samples_y
                 t+1,   # seed
                 None,
                 *scene_args)
    img = img / 256 # Normalize SDF to [0, 1]
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), 'results/single_circle_sdf/iter_{}.png'.format(t), gamma=2.2)
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    print('radius.grad:', radius_n.grad)
    print('center.grad:', center_n.grad)
    print('color.grad:', color.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current params.
    print('radius:', circle.radius)
    print('center:', circle.center)
    print('color:', circle_group.fill_color)

# Render the final result.
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)
img = render(256,   # width
             256,   # height
             2,     # num_samples_x
             2,     # num_samples_y
             102,    # seed
             None,
             *scene_args)
img = img / 256 # Normalize SDF to [0, 1]
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), 'results/single_circle_sdf/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_circle_sdf/iter_%d.png", "-vb", "20M",
    "results/single_circle_sdf/out.mp4"])