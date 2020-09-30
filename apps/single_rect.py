import pydiffvg
import torch
import skimage
import numpy as np

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 256 ,256
rect = pydiffvg.Rect(p_min = torch.tensor([40.0, 40.0]),
                     p_max = torch.tensor([160.0, 160.0]))
shapes = [rect]
rect_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))
shape_groups = [rect_group]
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
pydiffvg.imwrite(img.cpu(), 'results/single_rect/target.png', gamma=2.2)
target = img.clone()

# Move the rect to produce initial guess
# normalize p_min & p_max for easier learning rate
p_min_n = torch.tensor([80.0 / 256.0, 20.0 / 256.0], requires_grad=True)
p_max_n = torch.tensor([100.0 / 256.0, 60.0 / 256.0], requires_grad=True)
color = torch.tensor([0.3, 0.2, 0.5, 1.0], requires_grad=True)
rect.p_min = p_min_n * 256
rect.p_max = p_max_n * 256
rect_group.fill_color = color
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None, # background_image
             *scene_args)
pydiffvg.imwrite(img.cpu(), 'results/single_rect/init.png', gamma=2.2)

# Optimize for radius & center
optimizer = torch.optim.Adam([p_min_n, p_max_n, color], lr=1e-2)
# Run 100 Adam iterations.
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    rect.p_min = p_min_n * 256
    rect.p_max = p_max_n * 256
    rect_group.fill_color = color
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
    pydiffvg.imwrite(img.cpu(), 'results/single_rect/iter_{}.png'.format(t), gamma=2.2)
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    print('p_min.grad:', p_min_n.grad)
    print('p_max.grad:', p_max_n.grad)
    print('color.grad:', color.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current params.
    print('p_min:', rect.p_min)
    print('p_max:', rect.p_max)
    print('color:', rect_group.fill_color)

# Render the final result.
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
pydiffvg.imwrite(img.cpu(), 'results/single_rect/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_rect/iter_%d.png", "-vb", "20M",
    "results/single_rect/out.mp4"])
