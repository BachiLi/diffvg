import pydiffvg_tensorflow as pydiffvg
import tensorflow as tf
import skimage
import numpy as np

canvas_width, canvas_height = 256, 256
num_control_points = tf.constant([2, 2, 2])

points = tf.constant([[120.0,  30.0], # base
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
                     is_closed = True)
shapes = [path]
path_group = pydiffvg.ShapeGroup( shape_ids = tf.constant([0], dtype=tf.int32),
    fill_color = tf.constant([0.3, 0.6, 0.3, 1.0]))
shape_groups = [path_group]
scene_args = pydiffvg.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
render = pydiffvg.render
img = render(tf.constant(256), # width
             tf.constant(256), # height
             tf.constant(2),   # num_samples_x
             tf.constant(2),   # num_samples_y
             tf.constant(0),   # seed
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img, 'results/single_curve_tf/target.png', gamma=2.2)
target = tf.identity(img)

# Move the path to produce initial guess
# normalize points for easier learning rate
points_n = tf.Variable([[100.0/256.0,  40.0/256.0],  # base
                        [155.0/256.0,  65.0/256.0],  # control point
                        [100.0/256.0, 180.0/256.0],  # control point
                        [ 65.0/256.0, 238.0/256.0],  # base
                        [100.0/256.0, 200.0/256.0],  # control point
                        [170.0/256.0,  55.0/256.0],  # control point
                        [220.0/256.0, 100.0/256.0],  # base
                        [210.0/256.0,  80.0/256.0],  # control point
                        [140.0/256.0,  60.0/256.0]]) # control point
                       
color = tf.Variable([0.3, 0.2, 0.5, 1.0])
path.points = points_n * 256
path_group.fill_color = color
scene_args = pydiffvg.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(tf.constant(256), # width
             tf.constant(256), # height
             tf.constant(2),   # num_samples_x
             tf.constant(2),   # num_samples_y
             tf.constant(1),   # seed
             *scene_args)
pydiffvg.imwrite(img, 'results/single_curve_tf/init.png', gamma=2.2)

optimizer = tf.compat.v1.train.AdamOptimizer(1e-2)

for t in range(100):
    print('iteration:', t)

    with tf.GradientTape() as tape:
        # Forward pass: render the image.
        path.points = points_n * 256
        path_group.fill_color = color
        # Important to use a different seed every iteration, otherwise the result
        # would be biased.
        scene_args = pydiffvg.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(tf.constant(256), # width
                     tf.constant(256), # height
                     tf.constant(2),   # num_samples_x
                     tf.constant(2),   # num_samples_y
                     tf.constant(t+1), # seed,
                     *scene_args)
        loss_value = tf.reduce_sum(tf.square(img - target))

    print(f"loss_value: {loss_value}")
    pydiffvg.imwrite(img, 'results/single_curve_tf/iter_{}.png'.format(t))

    grads = tape.gradient(loss_value, [points_n, color])
    print(grads)
    optimizer.apply_gradients(zip(grads, [points_n, color]))

# Render the final result.
path.points = points_n * 256
path_group.fill_color = color
scene_args = pydiffvg.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(tf.constant(256), # width
             tf.constant(256), # height
             tf.constant(2),   # num_samples_x
             tf.constant(2),   # num_samples_y
             tf.constant(101),   # seed
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img, 'results/single_curve_tf/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_curve_tf/iter_%d.png", "-vb", "20M",
    "results/single_curve_tf/out.mp4"])
