import pydiffvg
import torch
import skimage
import numpy as np

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 256, 256
# https://www.w3schools.com/graphics/svg_polygon.asp
points = torch.tensor([[120.0,  30.0],
                       [ 60.0, 218.0],
                       [210.0,  98.0],
                       [ 30.0,  98.0],
                       [180.0, 218.0]])
polygon = pydiffvg.Polygon(points = points, is_closed = True)
shapes = [polygon]
polygon_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                    fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))
shape_groups = [polygon_group]
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
pydiffvg.imwrite(img.cpu(), 'results/single_polygon/target.png', gamma=2.2)
target = img.clone()

# Move the polygon to produce initial guess
# normalize points for easier learning rate
points_n = torch.tensor([[140.0 / 256.0,  20.0 / 256.0],
                         [ 65.0 / 256.0, 228.0 / 256.0],
                         [215.0 / 256.0, 100.0 / 256.0],
                         [ 35.0 / 256.0,  90.0 / 256.0],
                         [160.0 / 256.0, 208.0 / 256.0]], requires_grad=True)
color = torch.tensor([0.3, 0.2, 0.5, 1.0], requires_grad=True)
polygon.points = points_n * 256
polygon_group.color = color
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None, # background_image
             *scene_args)
pydiffvg.imwrite(img.cpu(), 'results/single_polygon/init.png', gamma=2.2)

# Optimize for radius & center
optimizer = torch.optim.Adam([points_n, color], lr=1e-2)
# Run 100 Adam iterations.
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    polygon.points = points_n * 256
    polygon_group.color = color
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
    pydiffvg.imwrite(img.cpu(), 'results/single_polygon/iter_{}.png'.format(t), gamma=2.2)
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    print('points_n.grad:', points_n.grad)
    print('color.grad:', color.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current params.
    print('points:', polygon.points)
    print('color:', polygon_group.fill_color)

# Render the final result.
polygon.points = points_n * 256
polygon_group.color = color
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256,   # width
             256,   # height
             2,     # num_samples_x
             2,     # num_samples_y
             102,    # seed
             None, # background_image
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), 'results/single_polygon/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_polygon/iter_%d.png", "-vb", "20M",
    "results/single_polygon/out.mp4"])
