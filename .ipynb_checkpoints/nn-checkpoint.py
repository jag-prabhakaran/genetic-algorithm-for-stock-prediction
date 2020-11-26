import tensorflow as tf

mnist = tf.keras.datasets.mnist # (dataset of 28x28 images of handwritten digits 0-9)
(x_train, y_train), (x_test, y_test) = mnist.load_data