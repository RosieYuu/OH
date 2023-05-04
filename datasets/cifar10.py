import tensorflow as tf
import tensorflow_datasets as tfds
from datasets.augment import Augment

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_cifar10(batch_size):
    data_dir = 'datasets/data'
    data = tfds.load(name='cifar10', data_dir=data_dir)

    aug = Augment()

    def map_augmentation(x):
        img_1 = aug(x['image'], training=True)
        img_2 = aug(x['image'], training=True)
        x['image1'] = img_1
        x['image2'] = img_2
        return x

    def map_dummy(x):
        x['image'] = aug(x['image'], training=False)
        return x

    ds_train = data['train'].repeat().shuffle(50000).map(map_augmentation, num_parallel_calls=AUTOTUNE).batch(
        batch_size).prefetch(AUTOTUNE)
    ds_gallery = data['train'].map(map_dummy, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    ds_test = data['test'].map(map_dummy, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    return ds_train, ds_gallery, ds_test
