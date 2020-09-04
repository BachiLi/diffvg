import tensorflow as tf

class PixelFilter:
    def __init__(self,
                 type,
                 radius = tf.constant(0.5)):
        self.type = type
        self.radius = radius
