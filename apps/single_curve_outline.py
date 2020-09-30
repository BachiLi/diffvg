import pydiffvg
import torch
import skimage

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 256, 256
num_control_points = torch.tensor([2, 2, 2])
points = torch.tensor([[120.0,  30.0], # base
                       [150.0,  60.0], # control point
                       [ 90.0, 198.0], # control point
                       [ 60.0, 218.0], # base
                       [ 90.0, 180.0], # control point
                       [200.0,  65.0], # control point
                       [210.0,  98.0], # base
                       [220.0,  70.0], # control point
                       [130.0,  55.0]]) # control point
path = pydiffvg.Path(num_control_points = num_control_points,
                     points = points,
                     is_closed = True,
                     stroke_width = torch.tensor(5.0))
shapes = [path]
path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]),
                                 stroke_color = torch.tensor([0.6, 0.3, 0.6, 0.8]))
shape_groups = [path_group]
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
pydiffvg.imwrite(img.cpu(), 'results/single_curve_outline/target.png', gamma=2.2)
target = img.clone()

# Move the path to produce initial guess
# normalize points for easier learning rate
points_n = torch.tensor([[100.0/256.0,  40.0/256.0], # base
                         [155.0/256.0,  65.0/256.0], # control point
                         [100.0/256.0, 180.0/256.0], # control point
                         [ 65.0/256.0, 238.0/256.0], # base
                         [100.0/256.0, 200.0/256.0], # control point
                         [170.0/256.0,  55.0/256.0], # control point
                         [220.0/256.0, 100.0/256.0], # base
                         [210.0/256.0,  80.0/256.0], # control point
                         [140.0/256.0,  60.0/256.0]], # control point
                        requires_grad = True) 
fill_color = torch.tensor([0.3, 0.2, 0.8, 1.0], requires_grad=True)
stroke_color = torch.tensor([0.4, 0.7, 0.5, 0.5], requires_grad=True)
stroke_width_n = torch.tensor(10.0 / 100.0, requires_grad=True)
path.points = points_n * 256
path.stroke_width = stroke_width_n * 100
path_group.fill_color = fill_color
path_group.stroke_color = stroke_color
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None, # background_image
             *scene_args)
pydiffvg.imwrite(img.cpu(), 'results/single_curve_outline/init.png', gamma=2.2)

# Optimize
optimizer = torch.optim.Adam([points_n, fill_color, stroke_color, stroke_width_n], lr=1e-2)
# Run 200 Adam iterations.
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    path.points = points_n * 256
    path.stroke_width = stroke_width_n * 100
    path_group.fill_color = fill_color
    path_group.stroke_color = stroke_color
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
    pydiffvg.imwrite(img.cpu(), 'results/single_curve_outline/iter_{}.png'.format(t), gamma=2.2)
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    print('points_n.grad:', points_n.grad)
    print('fill_color.grad:', fill_color.grad)
    print('stroke_color.grad:', stroke_color.grad)
    print('stroke_width.grad:', stroke_width_n.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current params.
    print('points:', path.points)
    print('fill_color:', path_group.fill_color)
    print('stroke_color:', path_group.stroke_color)
    print('stroke_width:', path.stroke_width)

# Render the final result.
path.points = points_n * 256
path.stroke_width = stroke_width_n * 100
path_group.fill_color = fill_color
path_group.stroke_color = stroke_color
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256,   # width
             256,   # height
             2,     # num_samples_x
             2,     # num_samples_y
             202,    # seed
             None, # background_image
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), 'results/single_curve_outline/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_curve_outline/iter_%d.png", "-vb", "20M",
    "results/single_curve_outline/out.mp4"])
