import pydiffvg_tensorflow as pydiffvg
import tensorflow as tf
import skimage
import numpy as np

canvas_width = 256
canvas_height = 256
circle = pydiffvg.Circle(radius = tf.constant(40.0),
                         center = tf.constant([128.0, 128.0]))
shapes = [circle]
circle_group = pydiffvg.ShapeGroup(shape_ids = tf.constant([0], dtype = tf.int32),
    fill_color = tf.constant([0.3, 0.6, 0.3, 1.0]))
shape_groups = [circle_group]
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
pydiffvg.imwrite(img, 'results/single_circle_tf/target.png', gamma=2.2)
target = tf.identity(img)

# Move the circle to produce initial guess
# normalize radius & center for easier learning rate
radius_n = tf.Variable(20.0 / 256.0)
center_n = tf.Variable([108.0 / 256.0, 138.0 / 256.0])
color = tf.Variable([0.3, 0.2, 0.8, 1.0])
circle.radius = radius_n * 256
circle.center = center_n * 256
circle_group.fill_color = color
scene_args = pydiffvg.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(tf.constant(256), # width
             tf.constant(256), # height
             tf.constant(2),   # num_samples_x
             tf.constant(2),   # num_samples_y
             tf.constant(1),   # seed
             *scene_args)
pydiffvg.imwrite(img, 'results/single_circle_tf/init.png', gamma=2.2)

optimizer = tf.compat.v1.train.AdamOptimizer(1e-2)

for t in range(100):
    print('iteration:', t)

    with tf.GradientTape() as tape:
        # Forward pass: render the image.
        circle.radius = radius_n * 256
        circle.center = center_n * 256
        circle_group.fill_color = color
        # Important to use a different seed every iteration, otherwise the result
        # would be biased.
        scene_args = pydiffvg.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(tf.constant(256), # width
                     tf.constant(256), # height
                     tf.constant(2),   # num_samples_x
                     tf.constant(2),   # num_samples_y
                     tf.constant(t+1),   # seed,
                     *scene_args)
        loss_value = tf.reduce_sum(tf.square(img - target))

    print(f"loss_value: {loss_value}")
    pydiffvg.imwrite(img, 'results/single_circle_tf/iter_{}.png'.format(t))

    grads = tape.gradient(loss_value, [radius_n, center_n, color])
    print(grads)
    optimizer.apply_gradients(zip(grads, [radius_n, center_n, color]))

# Render the final result.
circle.radius = radius_n * 256
circle.center = center_n * 256
circle_group.fill_color = color
scene_args = pydiffvg.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(tf.constant(256), # width
             tf.constant(256), # height
             tf.constant(2),   # num_samples_x
             tf.constant(2),   # num_samples_y
             tf.constant(101),   # seed
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), 'results/single_circle_tf/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_circle_tf/iter_%d.png", "-vb", "20M",
    "results/single_circle_tf/out.mp4"])
