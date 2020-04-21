import numpy as np 
import h5py
from sklearn.model_selection import train_test_split

from utils import Config
from load_audio import load_audio


# Step 1: Loading the data from the folders, then storing them in numpy arrays
data_loader = load_audio()
print('Loading the data...')
features_english, labels_english = data_loader.load_all_files('english')
features_hindi, labels_hindi = data_loader.load_all_files('hindi')
features_mandarin, labels_mandarin = data_loader.load_all_files('mandarin')

features_train = np.concatenate((features_english, features_hindi, features_mandarin), axis=0)
labels_train = np.concatenate((labels_english, labels_hindi, labels_mandarin), axis=0)

X_train, X_test, Y_train, Y_test = train_test_split(features_train, labels_train, test_size=0.2, shuffle=True)

# Step 2: Store the mfcc data in an hdf5 file
filename = 'mfcc_data.hdf5'
print('Saving the data in an hdf5 file...')
with h5py.File(filename, 'w') as f:
    f['X_train'] = X_train
    f['X_test'] = X_test
    f['Y_train'] = Y_train
    f['Y_test'] = Y_test

