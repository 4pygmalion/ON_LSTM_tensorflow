import tensorflow as tf

# Equation (6)
def cumax(x, axis=-1):
    return tf.math.cumsum(tf.nn.softmax(x, axis), axis)
