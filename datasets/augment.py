from __future__ import absolute_import, division, print_function, unicode_literals

import random
import tensorflow as tf
import tensorflow_addons as tfa


def get_dataset_stat():
    image_size = 32
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    n_class = 10
    return image_size, mean, std, n_class


# noinspection PyDefaultArgument
class Augment(object):
    def __init__(self):
        self.brightness = .4
        self.contrast = .4
        self.hue = .4
        image_size, mean, std, n_class = get_dataset_stat()
        self.mean = mean
        self.std = std
        self.img_size = [image_size, image_size, 3]

    def _augment_test(self, x):
        x = self._resize(x)
        x = self._standardize(x)
        return x

    def _augmentv1(self, x, shape, coord=[[[0., 0., 1., 1.]]]):
        x = self._crop(x, shape, coord)
        x = self._resize(x)
        x = self._random_grayscale(x, p=.2)
        x = self._color_jitter(x)
        x = self._random_hflip(x)
        x = self._standardize(x)
        return x

    def _augmentv2(self, x, shape, radius, coord=[[[0., 0., 1., 1.]]]):
        x = self._crop(x, shape, coord)
        x = self._resize(x)
        x = self._random_color_jitter(x, p=.8)
        x = self._random_grayscale(x, p=.2)
        x = self._random_gaussian_blur(x, radius, p=.5)
        x = self._random_hflip(x)
        x = self._standardize(x)
        return x

    def _augment_lincls(self, x, shape, coord=[[[0., 0., 1., 1.]]]):
        x = self._crop(x, shape, coord)
        x = self._resize(x)
        x = self._standardize(x)
        return x

    def _standardize(self, x):
        x = tf.cast(x, tf.float32)
        return x

    @staticmethod
    def _crop(x, shape, coord=[[[0., 0., 1., 1.]]]):
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            image_size=shape,
            bounding_boxes=coord,
            area_range=(.5, 1.),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)

        offset_height, offset_width, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        x = tf.slice(x, [offset_height, offset_width, 0], [target_height, target_width, 3])
        return x

    def _resize(self, x):
        img_size = 224
        x = tf.image.resize(x, [img_size, img_size])
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _color_jitter(self, x, _jitter_idx=[0, 1, 2, 3]):
        random.shuffle(_jitter_idx)
        _jitter_list = [
            self._brightness,
            self._contrast,
            self._saturation,
            self._hue]
        for idx in _jitter_idx:
            # noinspection PyArgumentList
            x = _jitter_list[idx](x)
        return x

    def _random_color_jitter(self, x, p=.8):
        if tf.less(tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32), tf.cast(p, tf.float32)):
            x = self._color_jitter(x)
        return x

    def _brightness(self, x):
        x = tf.cast(x, tf.float32)
        delta = tf.random.uniform(
            shape=[],
            minval=1 - self.brightness,
            maxval=1 + self.brightness,
            dtype=tf.float32)

        x *= delta
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _contrast(self, x):
        x = tf.image.random_contrast(x, lower=max(0, 1 - self.contrast), upper=1 + self.contrast)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _saturation(self, x):
        x = tf.image.random_saturation(x, lower=max(0, 1 - self.contrast), upper=1 + self.contrast)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _hue(self, x):
        x = tf.image.random_hue(x, max_delta=self.hue)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _grayscale(self, x):
        return tf.image.rgb_to_grayscale(x)

    def _random_grayscale(self, x, p=.2):
        if tf.less(tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32), tf.cast(p, tf.float32)):
            x = self._grayscale(x)
            x = tf.tile(x, [1, 1, 3])
        return x

    def _random_hflip(self, x):
        return tf.image.random_flip_left_right(x)

    def _random_gaussian_blur(self, x, radius, p=.5):
        if tf.less(tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32), tf.cast(p, tf.float32)):
            x = tfa.image.gaussian_filter2d(x, filter_shape=radius)
        return x

    def __call__(self, x, training=True):
        if training:
            return self._augmentv1(x, self.img_size)
        else:
            return self._augment_test(x)
