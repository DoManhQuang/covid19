import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier


def load_data_train(_root, _name):
    return np.array(pd.read_csv(_root + '/' + _name, usecols=['chroma_stft',
                                             'rms',
                                             'spectral_centroid',
                                             'spectral_bandwidth',
                                             'spectral_rolloff',
                                             'zero_crossing_rate',
                                             'mfcc1',
                                             'mfcc2',
                                             'mfcc3',
                                             'mfcc4',
                                             'mfcc5',
                                             # 'mfcc6',
                                             # 'mfcc7',
                                             # 'mfcc8',
                                             # 'mfcc9',
                                             # 'mfcc10',
                                             # 'mfcc11',
                                             # 'mfcc12',
                                             # 'mfcc13',
                                             # 'mfcc14',
                                             # 'mfcc15',
                                             # 'mfcc16',
                                             # 'mfcc17',
                                             # 'mfcc18',
                                             # 'mfcc19',
                                             # 'mfcc20',
                                                              ])), \
           np.array(pd.read_csv(_root + '/' + _name, usecols=['label'])).T[0], \
           np.array(pd.read_csv(_root + '/' + _name, usecols=['uid'])).T[0]


def load_data_test(_root, _name):
    return np.array(pd.read_csv(_root + '/' + _name, usecols=['chroma_stft',
                                                              'rms',
                                                              'spectral_centroid',
                                                              'spectral_bandwidth',
                                                              'spectral_rolloff',
                                                              'zero_crossing_rate',
                                                              'mfcc1',
                                                              'mfcc2',
                                                              'mfcc3',
                                                              'mfcc4',
                                                              'mfcc5',
                                                              # 'mfcc6',
                                                              # 'mfcc7',
                                                              # 'mfcc8',
                                                              # 'mfcc9',
                                                              # 'mfcc10',
                                                              # 'mfcc11',
                                                              # 'mfcc12',
                                                              # 'mfcc13',
                                                              # 'mfcc14',
                                                              # 'mfcc15',
                                                              # 'mfcc16',
                                                              # 'mfcc17',
                                                              # 'mfcc18',
                                                              # 'mfcc19',
                                                              # 'mfcc20',
                                                              ]))


def kernel_svm(Xtrain, ylabel, Xtest):
    kernels = ['sigmoid', 'rbf', 'linear', 'poly']
    data = []
    for kernel in kernels:
        covid = svm.SVC(kernel=kernel, degree=3, gamma=4, coef0=0)
        covid.fit(Xtrain, ylabel)
        y_pred = covid.predict(Xtest)
        data.append(y_pred)
        # print(kernel)
        # print(y_pred)
    return data


def MLP_Classifier(X_data, y_label, X_test):
    clf = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
              solver='lbfgs')
    clf.fit(X_data, y_label)
    y_pred = clf.predict(X_test)
    return y_pred


if __name__ == '__main__':
    root = 'data_source'
    file_name_train = 'data-train-mfccs-mean.csv'
    file_name_test = 'data-test-mfccs-mean.csv'
    X_data, y_label, uid = load_data_train(root, file_name_train)
    X_test = load_data_test(root, file_name_test)








    pass