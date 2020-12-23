import tensorflow as tf
import math

class Circle:
    def __init__(self, radius, center, stroke_width = tf.constant(1.0), id = ''):
        self.radius = radius
        self.center = center
        self.stroke_width = stroke_width
        self.id = id

class Ellipse:
    def __init__(self, radius, center, stroke_width = tf.constant(1.0), id = ''):
        self.radius = radius
        self.center = center
        self.stroke_width = stroke_width
        self.id = id

class Path:
    def __init__(self, num_control_points, points, is_closed, stroke_width = tf.constant(1.0), id = '', use_distance_approx = False):
        self.num_control_points = num_control_points
        self.points = points
        self.is_closed = is_closed
        self.stroke_width = stroke_width
        self.id = id
        self.use_distance_approx = use_distance_approx

class Polygon:
    def __init__(self, points, is_closed, stroke_width = tf.constant(1.0), id = ''):
        self.points = points
        self.is_closed = is_closed
        self.stroke_width = stroke_width
        self.id = id

class Rect:
    def __init__(self, p_min, p_max, stroke_width = tf.constant(1.0), id = ''):
        self.p_min = p_min
        self.p_max = p_max
        self.stroke_width = stroke_width
        self.id = id

class ShapeGroup:
    def __init__(self,
                 shape_ids,
                 fill_color,
                 use_even_odd_rule = True,
                 stroke_color = None,
                 shape_to_canvas = tf.eye(3),
                 id = ''):
        self.shape_ids = shape_ids
        self.fill_color = fill_color
        self.use_even_odd_rule = use_even_odd_rule
        self.stroke_color = stroke_color
        self.shape_to_canvas = shape_to_canvas
        self.id = id
