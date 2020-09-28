import pydiffvg
import torch
import skimage

pydiffvg.set_print_timing(True)

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 256, 256
num_control_points = torch.tensor([2])
points = torch.tensor([[120.0,  30.0], # base
                       [150.0,  60.0], # control point
                       [ 90.0, 198.0], # control point
                       [ 60.0, 218.0]]) # base
thickness = torch.tensor([10.0, 5.0, 4.0, 20.0])
path = pydiffvg.Path(num_control_points = num_control_points,
                     points = points,
                     is_closed = False,
                     stroke_width = thickness)
shapes = [path]
path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = None,
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
pydiffvg.imwrite(img.cpu(), 'results/single_open_curve_thickness/target.png', gamma=2.2)
target = img.clone()

# Move the path to produce initial guess
# normalize points for easier learning rate
points_n = torch.tensor([[100.0/256.0,  40.0/256.0], # base
                         [155.0/256.0,  65.0/256.0], # control point
                         [100.0/256.0, 180.0/256.0], # control point
                         [ 65.0/256.0, 238.0/256.0]], # base
                        requires_grad = True) 
thickness_n = torch.tensor([10.0 / 100.0, 10.0 / 100.0, 10.0 / 100.0, 10.0 / 100.0],
                         requires_grad = True)
stroke_color = torch.tensor([0.4, 0.7, 0.5, 0.5], requires_grad=True)
path.points = points_n * 256
path.stroke_width = thickness_n * 100
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
pydiffvg.imwrite(img.cpu(), 'results/single_open_curve_thickness/init.png', gamma=2.2)

# Optimize
optimizer = torch.optim.Adam([points_n, thickness_n, stroke_color], lr=1e-2)
# Run 200 Adam iterations.
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    path.points = points_n * 256
    path.stroke_width = thickness_n * 100
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
    pydiffvg.imwrite(img.cpu(), 'results/single_open_curve_thickness/iter_{}.png'.format(t), gamma=2.2)
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    print('points_n.grad:', points_n.grad)
    print('thickness_n.grad:', thickness_n.grad)
    print('stroke_color.grad:', stroke_color.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current params.
    print('points:', path.points)
    print('thickness:', path.stroke_width)
    print('stroke_color:', path_group.stroke_color)

# Render the final result.
path.points = points_n * 256
path.stroke_width = thickness_n * 100
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
pydiffvg.imwrite(img.cpu(), 'results/single_open_curve_thickness/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_open_curve_thickness/iter_%d.png", "-vb", "20M",
    "results/single_open_curve_thickness/out.mp4"])
