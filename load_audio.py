import librosa
import os
import os.path as osp
from utils import Config, mapping
import numpy as np

class load_audio:
    def __init__(self):
        self.path = osp.join(Config['root_path'])
        self.sampling_rate = Config['sampling_rate']
        self.hop_time = Config['hop_time']  # in seconds, hop length
        self.fft_time = Config['fft_time']  # in seconds, seq length
        self.mfcc_features = Config['mfcc_features']
        print('Initializing data loader...\n')
        print('path...', self.path)
        print('sampling rate in Hz...', self.sampling_rate)
        print('hop time in s...', self.hop_time)
        print('fft time in s...', self.fft_time)
        print('number of mfcc features...', self.mfcc_features)
        print()


    def prepare_one_file(self, filename, label):
        y, sr = librosa.load(filename, sr=self.sampling_rate)
        print('Preparing File:', filename)
        mat = librosa.feature.mfcc (y=y, 
                                    sr=sr, 
                                    n_mfcc=self.mfcc_features, 
                                    n_fft=int(sr*self.fft_time), 
                                    hop_length=int(sr*self.hop_time))
        mat = mat.transpose()  # To get an Mx64 matrix
        M = mat.shape[0]
        label_vector = np.full((M, 1), label)
                
        return mat, label_vector

    def load_all_files(self, language):
        print('Loading files in...', language)
        directory = 'train_' + language
        directory = osp.join(self.path, directory)
        label = mapping[language]

        all_features = np.empty(shape=(0, self.mfcc_features))
        all_labels = np.zeros(shape=(0, 1))

        for filename in os.listdir(directory):
            path = osp.join(directory, filename)
            features, labels = self.prepare_one_file(path, label)

            all_labels = np.concatenate((all_labels, labels), axis=0)
            all_features = np.concatenate((all_features, features), axis=0)

        s = Config['sequence_length']
        num_features = all_features.shape[0]
        if num_features % s != 0:
            remove = num_features % s
            all_features = all_features[:-remove,:]        
        all_features = all_features.reshape((int(all_features.shape[0]/s), s, all_features.shape[1]))
        language_labels = np.zeros(shape=(3, ))
        language_labels[label] = 1
        all_labels = np.full((all_features.shape[0], s, 3), language_labels)
        return all_features, all_labels


