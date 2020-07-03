from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # 다섯 번째 에포크마다 가중치를 저장합니다
    period=5)

def train():
    model = create_model()
    model.save_weights(checkpoint_path.format(epoch=0))
    model.fit(train_images, train_labels,
              epochs=50, callbacks=[cp_callback],
              validation_data=(test_images, test_labels),
              verbose=0)

def test():
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model = create_model()
    model.load_weights(latest)
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print("복원된 모델의 정확도: {:5.2f}%".format(100 * acc))

test()