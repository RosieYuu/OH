import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(tf.test.is_gpu_available())

from datasets import cifar10
from models.ohnet import OHNet
from utils.eval_tools import eval_cls_map


# Hyperparameter
learning_rate = 1e-5
batch_size = 50
epochs = 200

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_loss = tf.keras.metrics.Mean(name='loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(model, image1, image2):
    with tf.GradientTape() as tape:
        logits1, labels1, logits2, labels2 = model(image1, image2, training=False)
        loss = loss_object(labels1, logits1) + loss_object(labels2, logits2)

    gradients = tape.gradient(loss, model.encoder_q.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.encoder_q.trainable_variables))
    train_loss.update_state(loss)
    train_accuracy.update_state(labels2, logits2)


@tf.function
def eval_step(model, image):
    b, c = model(image, training=False)
    return b, c


def train_model(model, dataset, save_path):
    ds_train, ds_gallery, ds_test = dataset

    best_metrics = []
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
        query = None
        target = None
        cls1 = None
        cls2 = None

        for index, data in enumerate(ds_train):
            if index >= 50000 / batch_size: break

            train_step(model, data['image1'], data['image2'])

            if index % 100 == 0:
                train_logs = '{} - Epoch: [{}][{}/{}]\t Loss: {}\t Acc: {}'
                print(train_logs.format('TRAIN', epoch + 1, index, 50000 / batch_size,
                                        train_loss.result(), train_accuracy.result()))

        if (epoch+1) % 10 == 0:
            for index, data in enumerate(ds_gallery):
                b_train, _ = eval_step(model.encoder_q, data['image'])
                target = b_train if index == 0 else tf.concat([target, b_train], axis=0)
                one_hot_label = tf.one_hot(tf.squeeze(data['label']), 10)
                cls2 = one_hot_label if index == 0 else tf.concat([cls2, one_hot_label], axis=0)

            for index, data in enumerate(ds_test):
                b_query, _ = eval_step(model.encoder_q, data['image'])
                query = b_query if index == 0 else tf.concat([query, b_query], axis=0)
                one_hot_label = tf.one_hot(tf.squeeze(data['label']), 10)
                cls1 = one_hot_label if index == 0 else tf.concat([cls1, one_hot_label], axis=0)

            mAP = eval_cls_map(query, target, cls1, cls2, topk=1000)
            print("mAP:{}".format(mAP))

        else:
            mAP = 0
            print()

        metrics = [mAP]
        if sum(metrics) > sum(best_metrics):
            print("saving", end="\n\n")
            best_metrics = metrics
            model.save_weights(save_path + "ckpt/checkpoints")

    print(best_metrics)


if __name__ == '__main__':
    save_path_freeze = "results/cifar10/"

    dataset = cifar10.load_cifar10(batch_size=batch_size)
    model = OHNet(input_shape=(None, 224, 224, 3), middle_dim=2048, b_dim=64, c_dim=1024)
    train_model(model, dataset, save_path_freeze)
    print("exit")
