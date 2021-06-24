import librosa.display
import numpy as np
import pandas as pd


def lib_chroma_stft(_y, _sr, _round):
    return np.round(np.mean(librosa.feature.chroma_stft(y=_y, sr=_sr)), _round)


def lib_chroma_stft_vector(_y, _sr, _round):
    chroma = librosa.feature.chroma_stft(y=_y, sr=_sr)
    chroma -= np.mean(chroma, axis=0) + 1e-8
    return np.round(np.mean(chroma, 1), _round)


def lib_spectral_centroid(_y, _sr, _round):
    return np.round(np.mean(librosa.feature.spectral_centroid(y=_y, sr=_sr)[0]), _round)


def lib_rms(_y, _round):
    return np.round(np.mean(librosa.feature.rms(y=_y)[0]), _round)


def lib_spectral_bandwidth(_y, _sr, _round):
    return np.round(np.mean(librosa.feature.spectral_bandwidth(y=_y, sr=_sr)), _round)


def lib_spectral_rolloff(_y, _sr, _round):
    return np.round(np.mean(librosa.feature.spectral_rolloff(y=_y, sr=_sr)), _round)


def lib_zero_crossing_rate(_y, _round):
    return np.round(np.mean(librosa.feature.zero_crossing_rate(_y)[0]), _round)


def lib_mfcc(_y, _sr, _round):
    mfc = librosa.feature.mfcc(y=_y, sr=_sr, n_mfcc=20)
    mfc -= np.mean(mfc, axis=0) + 1e-8
    return np.round(np.mean(mfc, 1), _round)


def concat_lib(stft, rms, cent, band, roll, zero, mfcc):
    lib_row = [stft, rms, cent, band, roll, zero]
    for data in mfcc:
        lib_row.append(data)
    return lib_row


def data_row_lib(_x, _sr, _round):
    return concat_lib(lib_chroma_stft(_x, _sr, _round),
                      lib_rms(_x, _round),
                      lib_spectral_centroid(_x, _sr, _round),
                      lib_spectral_bandwidth(_x, _sr, _round),
                      lib_spectral_rolloff(_x, _sr, _round),
                      lib_zero_crossing_rate(_x, _round),
                      lib_mfcc(_x, _sr, _round))


def audio_data_wav(root, audio, name):
    return librosa.load(root + '/' + audio + '/' + name)


def save_to_csv(data, save_file):
    print(len(data[0]))
    df = pd.DataFrame(data, columns=['chroma_stft',
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
                                     'mfcc6',
                                     'mfcc7',
                                     'mfcc8',
                                     'mfcc9',
                                     'mfcc10',
                                     'mfcc11',
                                     'mfcc12',
                                     'mfcc13',
                                     'mfcc14',
                                     'mfcc15',
                                     'mfcc16',
                                     'mfcc17',
                                     'mfcc18',
                                     'mfcc19',
                                     'mfcc20',
                                     'label',
                                     ])
    df.to_csv(save_file, index=True, header=True, index_label='uid')
    print(df)
    pass


def load_data_csv(root, name):
    return np.array(pd.read_csv(root + '/' + name, usecols=['uuid', 'file_path', 'assessment_result']))


def get_data_file_name_and_label(dataframe):
    file_name = []
    label = []
    for data in dataframe:
        file_name.append(data[2])
        label.append(data[1])
    return np.array(file_name), np.array(label)


def get_data_lib(root, audio, filesname, label):
    data = []
    for i in range(0, len(filesname)):
        _x, _sr = audio_data_wav(root, audio, filesname[i])
        _round = 6
        data_row = data_row_lib(_x, _sr, _round)
        data_row.append(label[i])
        print(data_row)
        data.append(data_row)
    return data


def save_data_csv(_root, _audio, _file, _save_file):
    data_frame = load_data_csv(_root, _file)
    files_name, label = get_data_file_name_and_label(data_frame)
    data_set = get_data_lib(_root, _audio, files_name, label)
    save_to_csv(data_set, _save_file)
    pass


if __name__ == '__main__':
    # train
    root_train = 'aicv115m_public_train'
    audio_train = 'train_audio_files_8k'
    file_train = 'metadata_train_challenge.csv'
    save_file_train = 'data_source/data-train-mfccs-mean.csv'
    # test
    root_test = 'aicv115m_public_test'
    audio_test = 'public_test_audio_files_8k'
    file_test = 'metadata_public_test.csv'
    save_file_test = 'data_source/data-test-mfccs-mean.csv'

    save_data_csv(root_train, audio_train, file_train, save_file_train)
    save_data_csv(root_test, audio_test, file_test, save_file_test)





    pass