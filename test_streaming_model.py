import numpy as np 
import h5py
import os.path as osp
import os

from tensorflow.keras import Model
from tensorflow.keras.models import load_model

import librosa

from utils import Config
from load_audio import load_audio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

load_audio = load_audio()
streaming_model = load_model('streaming_model.hdf5')
plt.style.use('seaborn-colorblind')

seq_length = Config['sequence_length']

# Have an hdf5 file on hand with test samples in shape (num_samples, 500, 64)?
train_data = True

# Have a .wav file stored of which you want ot predict the languege?
predict_file = False

#########################################
# Using my Previously Created Test-File #
#########################################

# This function is run to import data from an hdf5 file
# Requirements: hdf5 file with X_train - training data in shape
# (num_samples, 500, 64)
# and Y_train - Training labels in shape (num_samples, 500, 3)
def use_train_data():

    # Importing the data from the hdf5 file
    filename = 'mfcc_data.hdf5'
    with h5py.File(filename, 'r') as f:
        X_test = f['X_test'][:]
        Y_test = f['Y_test'][:]

    num_samples = X_test.shape[0]

    # Create Confusion matrix, to illustrate results.
    confusion_matrix = np.zeros(shape=(3, 3))

    correct = 0
    false = 0

    for sample in range(num_samples):

        prediction = streaming_model.predict(X_test[sample].reshape(1, Config['sequence_length'], Config['mfcc_features']))

        pred_label = np.argmax(prediction[0, -1])
        actual_label = np.argmax(Y_test[sample][-1])

        if pred_label == actual_label:
            correct += 1
        else:
            false += 1

        confusion_matrix[actual_label][pred_label] += 1
        
        # Print small message every 100 samples
        if sample % 100 == 0:
            print('Sample:', sample, '...')

    axis = ['english', 'hindi', 'mandarin']

    # Normalize the confusion matrix to get percentages
    confusion_matrix = confusion_matrix/confusion_matrix.sum(axis=0)[:,None]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix, interpolation='nearest')
    fig.colorbar(cax)
    ax.set_xticklabels(['']+axis)
    ax.set_yticklabels(['']+axis)
    plt.xlabel('Actual Label')
    plt.ylabel('Predicted Label')
    plt.savefig('Images/confusion_matrix.eps')
    plt.show()
    plt.close()

    print('\n--- Results: ---\n')
    print('Correct:          {}'.format(correct))
    print('False:            {}'.format(false))
    print('Total Samples:    {}'.format(num_samples))
    print('Sums up?          {}'.format(correct + false == num_samples))
    print('Prediction Rate:  {}%\n'.format(round(100 * correct / num_samples, 3)))


if train_data:
    use_train_data()

#################################
# Predict label of unseen Audio #
#################################

# Modify to predict language of any audio file
directory = 'train/train_english'
filename = 'english_0094.wav'

path = osp.join(directory, filename)

# This function takes in one .wav file from a pre-defined location on
# the hard-drive, then predicts whether it is english, hindi or mandarin
def predict_one_file(path):
    
    # Get MFCC features and format the numpy array
    features, _ = load_audio.prepare_one_file(path, 0)
    features = features[:features.shape[0] - features.shape[0] % seq_length]
    features = features.reshape(int(features.shape[0] / seq_length), seq_length, Config['mfcc_features'])

    # Predicting the language
    num_seq = features.shape[0]
    all_pred = np.zeros(shape=3, )

    for seq in range(num_seq):
        prediction = streaming_model.predict(features[seq].reshape(1, Config['sequence_length'], Config['mfcc_features']))
        pred_label = prediction[0, -1]

        all_pred += pred_label

    # Normalizing to get the percentages
    all_pred = all_pred / num_seq

    # Printing out the Results
    print('\n--- Results: ---\n')
    print('Chances that it is English:   {}%'.format(round(100*all_pred[0], 3)))
    print('Chances that it is Hindi:     {}%'.format(round(100*all_pred[1], 3)))
    print('Chances that it is Mandarin:  {}%\n'.format(round(100*all_pred[2], 3)))


if predict_file:
    predict_one_file(path)


