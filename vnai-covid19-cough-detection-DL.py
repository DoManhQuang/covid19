import tensorflow as tf
from tensorflow.keras import layers, models


def created_model_cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5),
                            activation='relu', input_shape=(320, 240, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5),
                            activation='relu'))
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5),
                            activation='relu'))
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5),
                            activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.15))
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5),
                            activation='relu'))  # 3rd
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5),
                            activation='relu'))  # 4rd
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5),
                            activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.15))
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5),
                            activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(2, activation='softmax'))
    return model


if __name__ == '__main__':
    model_cnn = created_model_cnn()
    model_cnn.summary()
    model_cnn.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])














