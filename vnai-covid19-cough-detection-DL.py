import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


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
    # model = models.Sequential()
    # model.add(layers.Conv2D(filters=1, kernel_size=(5, 5), activation='relu', input_shape=(320, 240, 1),
    #                         padding='same'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same'))
    # model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same'))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.Dropout(0.15))
    # model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))
    # model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))
    # model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.Dropout(0.15))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(256, activation='relu'))
    # model.add(layers.Dropout(0.3))
    # model.add(layers.Dense(number_classify, activation='softmax'))
    model = models.Sequential()
    model.add(layers.MaxPooling2D(pool_size=(2, 2), input_shape=(240, 320, 1)))
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.15))
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.15))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(number_classify, activation='softmax'))
    return model


def save_csv(predicted_classes):
    res = {
        'assessment_result': predicted_classes,
    }
    df = pd.DataFrame(res, columns=['assessment_result'])
    df.to_csv('aicv115m_public_test/result_cnn.csv', index=True, header=True, index_label='uid')
    pass


def build_cnn_model_test(_root, _name, _local):
    data, label = get_data_images(_root, _name, _local)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    # print('Training data shape : ', x_train.shape, y_train.shape)
    # print('Testing data shape : ', x_test.shape, y_test.shape)

    number_classify = len(np.unique(y_train))  # so lop

    # chuan hoa du lieu
    x_train = x_train.reshape(-1, 240, 320, 1)
    x_test = x_test.reshape(-1, 240, 320, 1)
    # print('Training data shape : ', x_train.shape, y_train.shape)
    # print('Testing data shape : ', x_test.shape, y_test.shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.
    x_test = x_test / 255.

    # change for category label using one-hot encoding
    train_y_one_hot = to_categorical(y_train)
    test_y_one_hot = to_categorical(y_test)

    train_x, valid_x, train_label, valid_label = train_test_split(x_train, train_y_one_hot,
                                                                  test_size=0.2,
                                                                  random_state=10)

    model_cnn = created_model_cnn(number_classify)
    model_cnn.summary()
    model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    epochs = 25
    batch_size = 64

    model_train_covid = model_cnn.fit(train_x, train_label, batch_size=batch_size, epochs=epochs, verbose=1,
                                      validation_data=(valid_x, valid_label))
    # model_cnn.save(_root + "/model_cnn/CNN_MFC.h5py")

    test_eval = model_cnn.evaluate(x_test, test_y_one_hot, verbose=1)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])

    accuracy = model_train_covid.history['accuracy']
    val_accuracy = model_train_covid.history['val_accuracy']
    loss = model_train_covid.history['loss']
    val_loss = model_train_covid.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    predicted_classes = model_cnn.predict(x_test)
    predicted_classes = np.argmax(predicted_classes, axis=1)
    print(predicted_classes)

    target_names = ["Class {}".format(i) for i in range(number_classify)]
    print(classification_report(y_test, predicted_classes, target_names=target_names))


    pass


def load_model_cnn(_root, _name, _local):
    pass


if __name__ == '__main__':

    root_data = 'data_source'
    metadata = 'metadata_image_train.csv'
    local_image = 'data_source/MFCC_IMG_TRAIN/'
    build_cnn_model_test(root_data, metadata, local_image)

    pass













