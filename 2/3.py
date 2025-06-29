
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_num_capsules = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(
            shape=[self.input_num_capsules, self.num_capsules,
                   self.input_dim_capsule, self.dim_capsule],
            initializer='glorot_uniform',
            trainable=True,
            name='W'
        )

    def call(self, inputs):
        inputs_expanded = tf.expand_dims(tf.expand_dims(inputs, 1), 3)
        inputs_tiled = tf.tile(inputs_expanded, [1, self.num_capsules, 1, 1, 1])
        u_hat = tf.einsum('bijh, ijhd -> bijd', tf.squeeze(inputs_tiled, axis=-1), self.W)

        b = tf.zeros(shape=[tf.shape(inputs)[0], self.num_capsules, self.input_num_capsules])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            s = tf.reduce_sum(tf.multiply(c[:, :, :, tf.newaxis], u_hat), axis=2)
            v = self.squash(s)
            if i < self.routings - 1:
                b += tf.reduce_sum(u_hat * v[:, :, tf.newaxis, :], axis=-1)
        return v

    def squash(self, vector):
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -1, keepdims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm)
        scalar_factor /= tf.sqrt(vec_squared_norm + 1e-9)
        return scalar_factor * vector

class CapsuleNetWithGSA:
    def __init__(self, input_shape, n_class, routings):
        self.input_shape = input_shape
        self.n_class = n_class
        self.routings = routings

    def build_model(self):
        input_layer = layers.Input(shape=self.input_shape)
        conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1,
                              padding='valid', activation='relu')(input_layer)
        primary_caps = layers.Conv2D(filters=32*8, kernel_size=9, strides=2,
                                     padding='valid')(conv1)
        primary_caps = layers.Reshape(target_shape=[-1, 8])(primary_caps)
        primary_caps = layers.Lambda(self.squash)(primary_caps)
        digit_caps = CapsuleLayer(num_capsules=self.n_class, dim_capsule=16,
                                  routings=self.routings)(primary_caps)
        out_caps = layers.Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x), -1)))(digit_caps)

        model = models.Model(input_layer, out_caps)
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['accuracy'])
        return model

    def squash(self, x):
        squared_norm = tf.reduce_sum(tf.square(x), -1, keepdims=True)
        return squared_norm / (1 + squared_norm) * x / tf.sqrt(squared_norm + 1e-9)

    def train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
        model = self.build_model()
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test))
        return model, history

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    capsnet = CapsuleNetWithGSA(input_shape=(28, 28, 1), n_class=10, routings=3)
    model, history = capsnet.train(x_train, y_train, x_test, y_test, epochs=1)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")
