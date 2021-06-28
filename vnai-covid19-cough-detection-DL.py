import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_data_train(_root, _name):
    return np.array(pd.read_csv(_root + '/' + _name, usecols=['images'])), \
           np.array(pd.read_csv(_root + '/' + _name, usecols=['label'])).T[0]


def load_data_test(_root, _name):
    return np.array(pd.read_csv(_root + '/' + _name, usecols=['images']))


def get_image_320_240_gray(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)


def get_data_images(_root, _name, _local):
    image_data, label = load_data_train(_root, _name)
    data_images = []
    for name_image in image_data:
        path = _local + str(name_image[0])
        image_gray = get_image_320_240_gray(path)
        data_images.append(image_gray)
    return np.array(data_images), np.array(label)


def created_model_cnn(number_classify):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=1, kernel_size=(5, 5), activation='relu', input_shape=(320, 240, 1), padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.15))
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.15))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(number_classify, activation='softmax'))
    return model


def build_cnn_model_test(_root, _name, _local):
    data, label = get_data_images(_root, _name, _local)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    number_classify = len(np.unique(y_train))  # so lop

    # chuan hoa du lieu
    x_train = x_train.reshape(-1, 320, 240, 1)
    x_test = x_test.reshape(-1, 320, 240, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.
    x_test = x_test / 255.

    # change for category label using one-hot encoding
    train_y_one_hot = to_categorical(y_train)
    test_y_one_hot = to_categorical(y_test)

    train_x, valid_x, train_label, valid_label = train_test_split(x_train, train_y_one_hot,
                                                                  test_size=0.2,
                                                                  random_state=13)

    model_cnn = created_model_cnn(number_classify)
    model_cnn.summary()
    model_cnn.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
    epochs = 25
    batch_size = 64

    model_train_covid = model_cnn.fit(train_x, train_label, batch_size=batch_size, epochs=epochs, verbose=1,
                                      validation_data=(valid_x, valid_label))

    # accuracy = model_train_covid.history['acc']
    # val_accuracy = model_train_covid.history['val_acc']
    # loss = model_train_covid.history['loss']
    # val_loss = model_train_covid.history['val_loss']
    # epochs = range(len(accuracy))
    # plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    # plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    # plt.title('Training and validation accuracy')
    # plt.legend()
    # plt.figure()
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    # plt.show()
    #
    # predicted_classes = model_cnn.predict(x_test)
    # predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
    #
    # correct = np.where(predicted_classes == y_test)[0]
    # print("Found %d correct labels" % len(correct))
    # for i, correct in enumerate(correct[:1]):
    #     plt.subplot(3, 3, i + 1)
    #     plt.imshow(x_test[correct].reshape(320, 240), cmap='gray', interpolation='none')
    #     plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
    #     plt.tight_layout()
    #     plt.show()
    #
    # incorrect = np.where(predicted_classes != y_test)[0]
    # print("Found %d incorrect labels" % len(incorrect))
    # for i, incorrect in enumerate(incorrect[:1]):
    #     plt.subplot(3, 3, i + 1)
    #     plt.imshow(x_test[incorrect].reshape(320, 240), cmap='gray', interpolation='none')
    #     plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
    #     plt.tight_layout()
    #     plt.show()
    #
    # from sklearn.metrics import classification_report
    # target_names = ["Class {}".format(i) for i in range(number_classify)]
    # print(classification_report(y_test, predicted_classes, target_names=target_names))

    model_cnn.save(_root + "/model_cnn/CNN_model.h5py")
    pass


def load_model_cnn(_root, _name, _local):
    pass


if __name__ == '__main__':

    root_data = 'data_source'
    metadata = 'metadata_image_mfc_train.csv'
    local_image = 'data_source/mfc-image-train/'
    build_cnn_model_test(root_data, metadata, local_image)

    pass













