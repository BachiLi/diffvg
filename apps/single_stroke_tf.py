import pydiffvg_tensorflow as pydiffvg
import tensorflow as tf
import skimage
import numpy as np

canvas_width, canvas_height = 256, 256
num_control_points = tf.constant([2])

points = tf.constant([[120.0,  30.0], # base
                      [150.0,  60.0], # control point
                      [ 90.0, 198.0], # control point
                      [ 60.0, 218.0]]) # base
path = pydiffvg.Path(num_control_points = num_control_points,
                     points = points,
                     is_closed = False,
                     stroke_width = tf.constant(15.0))

shapes = [path]
path_group = pydiffvg.ShapeGroup( shape_ids = tf.constant([0], dtype=tf.int32),
                                  fill_color = tf.constant([0.0, 0.0, 0.0, 0.0]),
                                  stroke_color = tf.constant([0.6, 0.3, 0.6, 0.8]))
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
pydiffvg.imwrite(img, 'results/single_stroke_tf/target.png', gamma=2.2)
target = tf.identity(img)


# Move the path to produce initial guess
# normalize points for easier learning rate
points_n = tf.Variable([[100.0/256.0,  40.0/256.0], # base
                        [155.0/256.0,  65.0/256.0], # control point
                        [100.0/256.0, 180.0/256.0], # control point
                        [ 65.0/256.0, 238.0/256.0]] # base
                       ) 
stroke_color = tf.Variable([0.4, 0.7, 0.5, 0.5])
stroke_width_n = tf.Variable(5.0 / 100.0)
path.points = points_n * 256
path.stroke_width = stroke_width_n * 100
path_group.stroke_color = stroke_color
scene_args = pydiffvg.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(tf.constant(256), # width
             tf.constant(256), # height
             tf.constant(2),   # num_samples_x
             tf.constant(2),   # num_samples_y
             tf.constant(1),   # seed
             *scene_args)
pydiffvg.imwrite(img, 'results/single_stroke_tf/init.png', gamma=2.2)



optimizer = tf.compat.v1.train.AdamOptimizer(1e-2)

for t in range(100):
    print('iteration:', t)

    with tf.GradientTape() as tape:
        # Forward pass: render the image.
        path.points = points_n * 256
        path.stroke_width = stroke_width_n * 100
        path_group.stroke_color = stroke_color
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
    pydiffvg.imwrite(img, 'results/single_stroke_tf/iter_{}.png'.format(t))

    grads = tape.gradient(loss_value, [points_n, stroke_width_n, stroke_color])
    print(grads)
    optimizer.apply_gradients(zip(grads, [points_n, stroke_width_n, stroke_color]))


# Render the final result.
path.points = points_n * 256
path_group.stroke_color = stroke_color
scene_args = pydiffvg.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(tf.constant(256), # width
             tf.constant(256), # height
             tf.constant(2),   # num_samples_x
             tf.constant(2),   # num_samples_y
             tf.constant(101),   # seed
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img, 'results/single_stroke_tf/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_stroke_tf/iter_%d.png", "-vb", "20M",
    "results/single_curve_tf/out.mp4"])
