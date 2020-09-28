import pydiffvg
import torch
import skimage

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 510, 510
# https://www.flaticon.com/free-icon/black-plane_61212#term=airplane&page=1&position=8
shapes = pydiffvg.from_svg_path('M510,255c0-20.4-17.85-38.25-38.25-38.25H331.5L204,12.75h-51l63.75,204H76.5l-38.25-51H0L25.5,255L0,344.25h38.25l38.25-51h140.25l-63.75,204h51l127.5-204h140.25C492.15,293.25,510,275.4,510,255z')
path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))
shape_groups = [path_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)

render = pydiffvg.RenderFunction.apply
img = render(510, # width
             510, # height
             1,   # num_samples_x
             1,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
img = img / 510 # Normalize SDF to [0, 1]
pydiffvg.imwrite(img.cpu(), 'results/single_path_sdf/target.png', gamma=1.0)
target = img.clone()

# Move the path to produce initial guess
# normalize points for easier learning rate
noise = torch.FloatTensor(shapes[0].points.shape).uniform_(0.0, 1.0)
points_n = (shapes[0].points.clone() + (noise * 60 - 30)) / 510.0
points_n.requires_grad = True
color = torch.tensor([0.3, 0.2, 0.5, 1.0], requires_grad=True)
shapes[0].points = points_n * 510
path_group.fill_color = color
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)
img = render(510, # width
             510, # height
             1,   # num_samples_x
             1,   # num_samples_y
             1,   # seed
             None, # background_image
             *scene_args)
img = img / 510 # Normalize SDF to [0, 1]
pydiffvg.imwrite(img.cpu(), 'results/single_path_sdf/init.png', gamma=1.0)

# Optimize
optimizer = torch.optim.Adam([points_n, color], lr=1e-2)
# Run 100 Adam iterations.
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    shapes[0].points = points_n * 510
    path_group.fill_color = color
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        output_type = pydiffvg.OutputType.sdf)
    img = render(510,   # width
                 510,   # height
                 1,     # num_samples_x
                 1,     # num_samples_y
                 t+1,   # seed
                 None, # background_image
                 *scene_args)
    img = img / 510 # Normalize SDF to [0, 1]
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), 'results/single_path_sdf/iter_{}.png'.format(t), gamma=1.0)
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
    print('points:', shapes[0].points)
    print('color:', path_group.fill_color)

# Render the final result.
shapes[0].points = points_n * 510
path_group.fill_color = color
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)
img = render(510,   # width
             510,   # height
             1,     # num_samples_x
             1,     # num_samples_y
             102,    # seed
             None, # background_image
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), 'results/single_path_sdf/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_path_sdf/iter_%d.png", "-vb", "20M",
    "results/single_path_sdf/out.mp4"])
