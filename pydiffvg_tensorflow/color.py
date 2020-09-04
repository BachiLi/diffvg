import tensorflow as tf

class LinearGradient:
    def __init__(self,
                 begin = tf.constant([0.0, 0.0]),
                 end = tf.constant([0.0, 0.0]),
                 offsets = tf.constant([0.0]),
                 stop_colors = tf.constant([0.0, 0.0, 0.0, 0.0])):
        self.begin = begin
        self.end = end
        self.offsets = offsets
        self.stop_colors = stop_colors

class RadialGradient:
    def __init__(self,
                 center = tf.constant([0.0, 0.0]),
                 radius = tf.constant([0.0, 0.0]),
                 offsets = tf.constant([0.0]),
                 stop_colors = tf.constant([0.0, 0.0, 0.0, 0.0])):
        self.center = center
        self.radius = radius
        self.offsets = offsets
        self.stop_colors = stop_colors
