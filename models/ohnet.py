import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50


class Queue(object):
    def __init__(self, length, dim):
        self.dict = tf.Variable(tf.random.normal(shape=[length, dim]),
                                trainable=False)

    def update_queue(self, inputs):
        n = inputs.shape[0]
        update_dict = tf.concat([inputs, self.dict[:-n]], axis=0)
        self.dict.assign(update_dict)


class Encoder(Model):
    def __init__(self, middle_dim, b_dim, c_dim):
        super(Encoder, self).__init__()
        self.middle_dim = middle_dim
        self.b_dim = b_dim
        self.c_dim = c_dim

        self.backbone = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        self.backbone.trainable = True
        self.fc = Dense(units=middle_dim, activation='relu')
        self.b = Dense(units=b_dim, activation="sigmoid")
        self.c = Dense(units=c_dim)

    def call(self, inputs, **kwargs):
        feature = self.backbone(inputs)

        feature = self.fc(feature)
        b = self.b(feature)
        b = self.grad_through(hash=b)
        c = self.c(feature)
        c = tf.nn.l2_normalize(c, axis=-1)
        return b, c

    def grad_through(self, hash):
        binary_hash = tf.where(hash > 0.5, 1., 0.)
        return hash + tf.stop_gradient(binary_hash - hash)


class OHNet(Model):

    def __init__(self, input_shape, middle_dim, b_dim, c_dim, m=0.999, T=0.2, queue_lenght=4096):
        super(OHNet, self).__init__()
        self.middle_dim = middle_dim
        self.b_dim = b_dim
        self.c_dim = c_dim
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = Encoder(middle_dim, b_dim, c_dim)
        self.encoder_q.build(input_shape=input_shape)
        self.encoder_k = Encoder(middle_dim, b_dim, c_dim)
        self.encoder_k.build(input_shape=input_shape)

        # create the queue
        self.P_Queue = Queue(queue_lenght, c_dim)
        self.B_Queue = Queue(queue_lenght, b_dim)
        self.C_Queue = Queue(queue_lenght, c_dim)

        for param_q, param_k in zip(self.encoder_q.trainable_variables, self.encoder_k.trainable_variables):
            param_k.assign(param_q)

        self.softmax = tf.keras.layers.Activation('softmax')

    def update_queue(self, p, b, c):
        self.P_Queue.update_queue(p)
        self.B_Queue.update_queue(b)
        self.C_Queue.update_queue(c)

    # self-attention
    def contrastive(self, b, c):
        sim = tf.matmul(b, self.B_Queue.dict, transpose_b=True) + tf.matmul(1-b, 1-self.B_Queue.dict, transpose_b=True)
        p = tf.nn.l2_normalize(tf.matmul(self.softmax(sim / self.b_dim / self.T), self.C_Queue.dict), axis=-1) + 0.5 * c
        p = tf.nn.l2_normalize(p, axis=-1)
        return p

    def call_key_encoder(self, inputs):
        for param_q, param_k in zip(self.encoder_q.trainable_variables, self.encoder_k.trainable_variables):
            update_k = param_k * self.m + param_q * (1. - self.m)
            param_k.assign(update_k)

        idx_shuffle = tf.random.shuffle(range(inputs.shape[0]))
        inputs = tf.gather(inputs, idx_shuffle)

        b, c = self.encoder_k(inputs)

        idx_unshuffle = tf.argsort(idx_shuffle)
        b = tf.gather(b, idx_unshuffle)
        c = tf.gather(c, idx_unshuffle)

        k = self.contrastive(b, c)

        return tf.stop_gradient(k), tf.stop_gradient(b), tf.stop_gradient(c)

    def call_query_encoder(self, inputs):
        b, c = self.encoder_q(inputs)
        q = self.contrastive(b, c)
        return q, b, c

    def call(self, im_q, im_k):
        q, b_query, c_query = self.call_query_encoder(im_q)
        k, b_key, c_key = self.call_key_encoder(im_k)

        l_pos = tf.expand_dims(tf.einsum('nc,nc->n', q, k), axis=-1)
        l_neg = tf.einsum('nc,kc->nk', q, self.P_Queue.dict)
        logits1 = tf.concat([l_pos, l_neg], axis=-1)
        logits1 /= self.T
        labels1 = tf.zeros(logits1.shape[0])

        # dequeue and enqueue
        self.update_queue(k, b_key, c_key)

        q2, b_query2, c_query2 = self.call_query_encoder(im_k)
        q2 = tf.stop_gradient(q2)
        logits2 = tf.einsum('nc,kc->nk', q, q2)
        logits2 /= self.T
        labels2 = tf.range(logits2.shape[0])

        return logits1, labels1, logits2, labels2