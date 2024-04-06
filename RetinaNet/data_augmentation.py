import tensorflow as tf
import numpy as np
import math

# Define the size of the motion blur kernel
size = 3

# Create a zero-filled kernel matrix for motion blur
kernel_motion_blur = np.zeros((size, size))

# Set the central row of the kernel to ones, creating a motion blur effect
kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)

# Normalize the kernel by dividing by its size
kernel_motion_blur = kernel_motion_blur / size

# Expand the kernel to match the number of channels in the input image
kernel_motion_blur = np.expand_dims(kernel_motion_blur, axis=-1)
kernel_motion_blur = np.repeat(kernel_motion_blur, repeats=3, axis=-1)

# Expand the kernel's dimensions for compatibility with TensorFlow operations
kernel_motion_blur = np.expand_dims(kernel_motion_blur, axis=-1)

# Convert the kernel to float32 data type for TensorFlow operations
kernel_motion_blur = tf.cast(kernel_motion_blur, tf.float32)

# Function for flipping the input image horizontally with a certain probability
def random_flip_horizontal(image, boxes, prob=0.5):
    if tf.random.uniform(()) > prob:
        image = tf.image.flip_left_right(image)
        # Adjust bounding box coordinates to match the image flip
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes

# Function for randomly adjusting the contrast of the input image
def random_adjust_contrast(image, prob=0.5):
    if tf.random.uniform(()) > prob:
        # Randomly select a factor to adjust contrast within a range
        factor = tf.random.uniform((), 0.5, 2.0)
        return tf.image.adjust_contrast(image, factor)
    return image

# Function for randomly adjusting the brightness of the input image
def random_adjust_brightness(image, prob=0.5):
    if tf.random.uniform(()) > prob:
        return tf.image.random_brightness(image, 0.06)
    return image

# Function for generating a Gaussian blur kernel
def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)

# Function for applying random Gaussian blur to the input image
def random_gaussian_blur(img, prob=0.9):
    if tf.random.uniform(()) > prob:
        # Convert image to float32 for TensorFlow operations
        img = tf.cast(img, dtype=tf.float32)
        if tf.random.uniform(()) > 0.5:
            # Generate Gaussian blur kernel
            kernel = _gaussian_kernel(7, 3, 3, img.dtype)
        else:
            # Use pre-defined motion blur kernel
            kernel = kernel_motion_blur
        # Apply convolution operation for blurring
        img = tf.nn.depthwise_conv2d(img[None], kernel, [1, 1, 1, 1], "SAME")
        # Convert image back to uint8 data type
        return tf.cast(img[0], dtype=tf.uint8)
    return img
