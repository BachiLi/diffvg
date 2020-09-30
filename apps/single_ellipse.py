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
ellipse_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                    fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))
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
pydiffvg.imwrite(img.cpu(), 'results/single_ellipse/target.png', gamma=2.2)
target = img.clone()

# Move the ellipse to produce initial guess
# normalize radius & center for easier learning rate
radius_n = torch.tensor([20.0 / 256.0, 40.0 / 256.0], requires_grad=True)
center_n = torch.tensor([108.0 / 256.0, 138.0 / 256.0], requires_grad=True)
color = torch.tensor([0.3, 0.2, 0.8, 1.0], requires_grad=True)
ellipse.radius = radius_n * 256
ellipse.center = center_n * 256
ellipse_group.fill_color = color
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None, # background_image
             *scene_args)
pydiffvg.imwrite(img.cpu(), 'results/single_ellipse/init.png', gamma=2.2)

# Optimize for radius & center
optimizer = torch.optim.Adam([radius_n, center_n, color], lr=1e-2)
# Run 50 Adam iterations.
for t in range(50):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    ellipse.radius = radius_n * 256
    ellipse.center = center_n * 256
    ellipse_group.fill_color = color
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
    pydiffvg.imwrite(img.cpu(), 'results/single_ellipse/iter_{}.png'.format(t), gamma=2.2)
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
    print('radius:', ellipse.radius)
    print('center:', ellipse.center)
    print('color:', ellipse_group.fill_color)

# Render the final result.
ellipse.radius = radius_n * 256
ellipse.center = center_n * 256
ellipse_group.fill_color = color
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
pydiffvg.imwrite(img.cpu(), 'results/single_ellipse/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_ellipse/iter_%d.png", "-vb", "20M",
    "results/single_ellipse/out.mp4"])
