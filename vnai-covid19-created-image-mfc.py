import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def lib_mfc_mean_matrix(_y, _sr, _round):
    mfc = librosa.feature.mfcc(y=_y, sr=_sr, n_mfcc=10, lifter=6)
    mfc -= np.mean(mfc, axis=0) + 1e-8
    return mfc


def show_image_mfc(_mfc):
    plt.rcParams["figure.figsize"] = (3.2, 2.4)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(_mfc, ax=ax)
    plt.xlabel('time')
    plt.ylabel('frequency')
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC')
    plt.show()
    pass


def save_image(_mfc, _path):
    plt.rcParams["figure.figsize"] = (3.2, 2.4)
    fig, ax = plt.subplots()
    librosa.display.specshow(_mfc, ax=ax)
    plt.savefig(_path)
    plt.cla()
    plt.clf()
    plt.close('all')
    pass


def load_data_csv(root, name, _col):
    return np.array(pd.read_csv(root + '/' + name, usecols=_col))


def lib_audio_data_wav(root, audio, name):
    return librosa.load(root + '/' + audio + '/' + name)


def get_data_file_name_and_label(dataframe):
    file_name = []
    label = []
    for data in dataframe:
        file_name.append(data[2])
        label.append(data[1])
    return np.array(file_name), np.array(label)


def get_data_file_name(dataframe):
    file_name = []
    for data in dataframe:
        file_name.append(data[1])
    return np.array(file_name)


def save_data_image_mfc_test(_root, _audio, _file, _local, _col):
    data_frame = load_data_csv(_root, _file, _col)
    files_name = get_data_file_name(data_frame)
    _round = 6
    size = len(files_name)
    for i in range(0, size):
        _x, _sr = lib_audio_data_wav(_root, _audio, files_name[i])
        lib_mfc = lib_mfc_mean_matrix(_x, _sr, _round)
        save_path = _local + 'image_' + str(i) + '.png'
        save_image(lib_mfc, save_path)
        print(save_path)
        pass


def save_data_image_mfc_train(_root, _audio, _file, _local, _col):
    data_frame = load_data_csv(_root, _file, _col)
    files_name, _label = get_data_file_name_and_label(data_frame)
    _round = 6
    size = len(files_name)
    for i in range(0, size):
        _x, _sr = lib_audio_data_wav(_root, _audio, files_name[i])
        lib_mfc = lib_mfc_mean_matrix(_x, _sr, _round)
        save_path = _local + 'image_' + str(i) + 'label_' + str(_label[i]) + '.png'
        save_image(lib_mfc, save_path)
        print(save_path)
        pass


def save_metadata_image_mfc_train(_root, _file, _col):
    data_frame = load_data_csv(_root, _file, _col)
    files_name, _label = get_data_file_name_and_label(data_frame)
    size = len(files_name)
    data_path = []
    for i in range(0, size):
        save_path = 'image_' + str(i) + 'label_' + str(_label[i]) + '.png'
        data_path.append(save_path)
        pass
    metadata_image_mfc = {
        'images': data_path,
        'label': _label,
    }
    df = pd.DataFrame(metadata_image_mfc, columns=['images', 'label'])
    df.to_csv('data_source/metadata_image_mfc_train.csv', index=True, header=True, index_label='uid')
    return df


def save_metadata_image_mfc_test(_root, _file, _col):
    data_frame = load_data_csv(_root, _file, _col)
    files_name = get_data_file_name(data_frame)
    size = len(files_name)
    data_path = []
    label = []
    for i in range(0, size):
        save_path = 'image_' + str(i) + '.png'
        data_path.append(save_path)
        label.append(0)

    metadata_image_mfc = {
        'images': data_path,
        'assessment_result': label,
    }
    df = pd.DataFrame(metadata_image_mfc, columns=['images', 'assessment_result'])
    df.to_csv('data_source/metadata_image_mfc_test.csv', index=True, header=True, index_label='uid')
    return df


if __name__ == '__main__':

    root_train = 'aicv115m_public_train'
    audio_train = 'train_audio_files_8k'
    file_train = 'metadata_train_challenge.csv'
    local_save_train = 'data_source/mfc-image-train/'
    use_col_train = ['uuid', 'file_path', 'assessment_result']

    root_test = 'aicv115m_public_test'
    audio_test = 'public_test_audio_files_8k'
    file_test = 'metadata_public_test.csv'
    local_save_test = 'data_source/mfc-image-test/'
    use_col_test = ['uuid', 'file_path']

    # save_data_image_mfc_test(root_test, audio_test, file_test, local_save_test, use_col_test)
    # save_data_image_mfc_train(root_train, audio_train, file_train, local_save_train, use_col_train)
    dfExcel_metadata_train = save_metadata_image_mfc_train(root_train, file_train, use_col_train)
    print(dfExcel_metadata_train)

    dfExcel_metadata_test = save_metadata_image_mfc_test(root_test, file_test, use_col_test)
    print(dfExcel_metadata_test)



    pass