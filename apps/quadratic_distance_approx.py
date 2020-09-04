import pydiffvg
import torch
import skimage
import numpy as np
import matplotlib.pyplot as plt

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 256, 256
num_control_points = torch.tensor([1])
points = torch.tensor([[ 50.0,  30.0], # base
                       [125.0, 400.0], # control point
                       [170.0,  30.0]]) # base
path = pydiffvg.Path(num_control_points = num_control_points,
                     points = points,
                     stroke_width = torch.tensor([30.0]),
                     is_closed = False,
                     use_distance_approx = False)
shapes = [path]
path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = None,
                                 stroke_color = torch.tensor([0.5, 0.5, 0.5, 0.5]))
shape_groups = [path_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)
render = pydiffvg.RenderFunction.apply
img = render(256, # width
             256, # height
             1,   # num_samples_x
             1,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
img /= 256.0
cm = plt.get_cmap('viridis')
img = cm(img.squeeze())
pydiffvg.imwrite(img, 'results/quadratic_distance_approx/ref_sdf.png')

scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
pydiffvg.imwrite(img, 'results/quadratic_distance_approx/ref_color.png')

shapes[0].use_distance_approx = True
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)
img = render(256, # width
             256, # height
             1,   # num_samples_x
             1,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
img /= 256.0
img = cm(img.squeeze())
pydiffvg.imwrite(img, 'results/quadratic_distance_approx/approx_sdf.png')

scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
pydiffvg.imwrite(img, 'results/quadratic_distance_approx/approx_color.png')