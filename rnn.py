import numpy as np 
import matplotlib.pyplot as plt 
import h5py

from tensorflow.keras.layers import Input, Dense, GRU, LSTM
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from sklearn.utils.class_weight import compute_class_weight

from utils import Config
from load_audio import load_audio

import librosa

plt.style.use('seaborn')

filename = 'mfcc_data.hdf5'

print('Loading the data...')

# Load the data from the hdf5 file
with h5py.File(filename, 'r') as f:
    X_train = f['X_train'][:]
    Y_train = f['Y_train'][:]

# Creating the model
training_input_shape = (Config['sequence_length'], Config['mfcc_features'])  # (500, 64)
training_input = Input(shape=training_input_shape, name='training_input')
x = LSTM(64, return_sequences=True, stateful=False, name='training_layer_1')(training_input)
x = LSTM(32, return_sequences=True, stateful=False, name='training_layer_2')(x)
x = Dense(32, activation='relu', name='training_layer_3')(x)
training_prediction = Dense(3, activation='softmax', name='training_output')(x)

training_model = Model(inputs=training_input, outputs=training_prediction)

training_model.compile(loss='categorical_crossentropy', 
                       optimizer='adam', 
                       metrics=['accuracy'])
training_model.summary()

results = training_model.fit(X_train, Y_train, batch_size=Config['batch_size'], 
                                               epochs=Config['epochs'], 
                                               validation_split=0.2, 
                                               shuffle=True,
                                               verbose=1)

training_model.save('training_model.hdf5')


# Creating the streaming model
streaming_input_shape = (1, None, Config['mfcc_features']) # (1, None, 64)
streaming_input = Input(batch_shape=streaming_input_shape, name='streaming_input') 
x = LSTM(64, return_sequences=True, stateful=True, name='streaming_layer_1')(streaming_input)
x = LSTM(32, return_sequences=True, stateful=False, name='streaming_layer_2')(x)
x = Dense(32, activation='relu', name='streaming_layer_3')(x)
streaming_pred = Dense(3, activation='softmax', name='streaming_output')(x)

streaming_model = Model(inputs=streaming_input, outputs=streaming_pred)

streaming_model.compile(loss='categorical_crossentropy', optimizer='adam')
streaming_model.summary()


# Copy the weights from trained model to streaming-inference model
training_model.save('training_model.hdf5')
streaming_model.load_weights('training_model.hdf5')


# Save streaming model and plots of both models - plots should be identical
streaming_model.save('streaming_model.hdf5')
plot_model(training_model, to_file='Images/training_model.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=128)
plot_model(streaming_model, to_file='Images/streaming_model.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=128)


# Plotting Accuracy and Loss
loss = results.history['loss']
val_loss = results.history['val_loss']
accuracy = results.history['accuracy']
val_accuracy = results.history['val_accuracy']

epochs = np.arange(len(accuracy))

# Plot the accuracy
plt.plot(epochs, accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy of the RNN model after each epoch')
plt.legend()
plt.ylim(0, 1.2)
plt.savefig('Images/Accuracy.eps')
plt.show()
plt.close()

# PLot the Loss
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss of the RNN model after each epoch')
plt.legend()
plt.savefig('Images/Loss.eps')
plt.show()
plt.close()


