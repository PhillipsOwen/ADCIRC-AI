import tensorflow as tf
# import tensorflow_hub as hub

print("TensorFlow version:", tf.__version__)
# print("TensorFlow Hub version:", hub.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
