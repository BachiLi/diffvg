import pydiffvg
import torch
import skimage
import numpy as np

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 256, 256
ellipse = pydiffvg.Ellipse(radius = torch.tensor([60.0, 30.0]),
                           center = torch.tensor([128.0, 128.0]))
shapes = [ellipse]
ellipse_group = pydiffvg.ShapeGroup(\
    shape_ids = torch.tensor([0]),
    fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]),
    shape_to_canvas = torch.eye(3, 3))
shape_groups = [ellipse_group]
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
pydiffvg.imwrite(img.cpu(), 'results/single_ellipse_transform/target.png', gamma=2.2)
target = img.clone()

# Affine transform the ellipse to produce initial guess
color = torch.tensor([0.3, 0.2, 0.8, 1.0], requires_grad=True)
affine = torch.zeros(2, 3)
affine[0, 0] = 1.3
affine[0, 1] = 0.2
affine[0, 2] = 0.1
affine[1, 0] = 0.2
affine[1, 1] = 0.6
affine[1, 2] = 0.3
affine.requires_grad = True
shape_to_canvas = torch.cat((affine, torch.tensor([[0.0, 0.0, 1.0]])), axis=0)
ellipse_group.fill_color = color
ellipse_group.shape_to_canvas = shape_to_canvas
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None, # background_image
             *scene_args)
pydiffvg.imwrite(img.cpu(), 'results/single_ellipse_transform/init.png', gamma=2.2)

# Optimize for radius & center
optimizer = torch.optim.Adam([color, affine], lr=1e-2)
# Run 150 Adam iterations.
for t in range(150):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    ellipse_group.fill_color = color
    ellipse_group.shape_to_canvas = torch.cat((affine, torch.tensor([[0.0, 0.0, 1.0]])), axis=0)
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
    pydiffvg.imwrite(img.cpu(), 'results/single_ellipse_transform/iter_{}.png'.format(t), gamma=2.2)
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    print('color.grad:', color.grad)
    print('affine.grad:', affine.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current params.
    print('color:', ellipse_group.fill_color)
    print('affine:', affine)

# Render the final result.
ellipse_group.fill_color = color
ellipse_group.shape_to_canvas = torch.cat((affine, torch.tensor([[0.0, 0.0, 1.0]])), axis=0)
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
pydiffvg.imwrite(img.cpu(), 'results/single_ellipse_transform/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_ellipse_transform/iter_%d.png", "-vb", "20M",
    "results/single_ellipse_transform/out.mp4"])
