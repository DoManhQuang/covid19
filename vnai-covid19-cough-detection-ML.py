import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def load_data_train(_root, _name):
    return np.array(pd.read_csv(_root + '/' + _name, usecols=['images'])), \
           np.array(pd.read_csv(_root + '/' + _name, usecols=['label'])).T[0]


def load_data_test(_root, _name):
    return np.array(pd.read_csv(_root + '/' + _name, usecols=['images']))


def get_pca_vector(image_path, components):
    img = cv2.imread(image_path)  # you can use any image you want.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img_gray.shape)
    pca = PCA(n_components=components)
    pca.fit(img_gray)
    # print(pca.explained_variance_ratio_)
    # plt.imshow(img_gray)
    # plt.show()
    return pca.explained_variance_ratio_


def get_data_mfc_pca(root, name, local, comp):
    image_data, label = load_data_train(root, name)
    # print(data[0][0])
    data_mfc_pca = []
    for image in image_data:
        path = local + str(image[0])
        # print(path)
        pca_vector = get_pca_vector(path, comp)
        # print(pca_vector)
        data_mfc_pca.append(pca_vector)
    return data_mfc_pca, label


def get_data_mfc_pca_submit_test(root, name, local, comp):
    image_data = load_data_test(root, name)
    # print(data[0][0])
    data_mfc_pca = []
    for image in image_data:
        path = local + str(image[0])
        # print(path)
        pca_vector = get_pca_vector(path, comp)
        # print(pca_vector)
        data_mfc_pca.append(pca_vector)
    return data_mfc_pca


def mfc_pca_svm_model(root, name, local):
    comp = 6
    data_mfc_pca, label = get_data_mfc_pca(root, name, local, comp)
    # print(data_mfc_pca, label)
    for i in range(0, 10):
        X_train, X_test, y_train, y_test = train_test_split(data_mfc_pca, label, test_size=0.2)
        # print(X_train)
        # print(y_test)
        clf = SVC(kernel='rbf', gamma=0.1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        res = []
        res.append(100 * accuracy_score(y_test, y_pred))
        print(str(i) + "/ RBF Accuracy: %.2f %%" % (100 * accuracy_score(y_test, y_pred)))
        plot_confusion_matrix(clf, X_test, y_test)
        plt.show()
    print('MEAN: ', np.mean(res))


def save_csv_result(root_train, name_train, local_train, root_test, name_test, local_test):
    comp = 2
    X_train, y_train = get_data_mfc_pca(root_train, name_train, local_train, comp)
    X_test = get_data_mfc_pca_submit_test(root_test, name_test, local_test, comp)
    # print(X_test)
    clf = SVC(kernel='rbf',  gamma=0.01)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metadata_image_mfc = {
        'assessment_result': y_pred,
    }
    df = pd.DataFrame(metadata_image_mfc, columns=['assessment_result'])
    df.to_csv('aicv115m_public_test/RBF_result_svm_pca_mfc.csv', index=True, header=True, index_label='uid')
    pass


if __name__ == '__main__':
    root_data = 'data_source'
    metadata = 'metadata_image_mfc_train.csv'
    local_image = 'data_source/mfc-image-train/'

    metadata_test = 'metadata_image_mfc_test.csv'
    local_image_test = 'data_source/mfc-image-test/'

    # mfc_pca_svm_model(root_data, metadata, local_image)
    save_csv_result(root_data, metadata, local_image, root_data, metadata_test, local_image_test)



