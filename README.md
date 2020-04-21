# Language Identification Algorithm in Python

*This is my very first repository, please don't be too harsh on it!*

## Project Description

This project tries to identify the language of spoken audio and classify it into English, Hindi and Mandarin using a Recurrent Neural Network (RNN) in Python. I used three programs that would do three separate tasks and are supposed to be run in order. In addition, I have 2 helper files, one has a class, the other one the parameters that are being used.

## Loading The Data

In any Machine Learning Project, it is always important to use data correctly. In this project, I have .wav files of people speaking English, Hindi and Mandarin that I will use to train my model. The file **dataset.py** does that with the help of the class load\_audio in **load\_audio.py**. The files I had were stored in the folder *train/train_\** (where * denotes the language).

The class load\_audio reads in all *.wav* files, slices them into small samples of 10ms, then extracs 64 mel frequency cepstral coefficients (or MFCC). I decided on a sequence lenght of 5 seconds, so 500 of these samples will make up one sequence. Each sequence will get the label according to its language:

* English: 0
* Hindi: 1
* Mandarin: 2

In the end, the data is split into 80% training, 20% testing and is stored in an hdf5 file.

## Training The Model

The next program I created is called **rnn.py** and it trains the model. First, it reads in the hdf5 file we stored in the beginning and imports the training data. I decided on using 2 Long Short Term Memory (LSTM) layers and one Dense layers as my hidden layers, a batch size of 10 sequences and 30 epochs. Those parameters and others I stored in a dictionary called Config in **utils.py**.

After training, I saved the model in an hdf5 file.

## Testing the model

When I prepared my data in the beginning, I set aside 20% for testing. So I created a file called **test\_streaming\_model.py** to do that. Also, it might be useful to have the model and see how it can predict on any unseen audio. Assuming, we have a .wav file and we would like to know the language of it. Thus, I created a function that takes in a path to a file, then prints out the probabilities on which language it is.

## Some Notes

* **Accuracy** I tracked the categorical accuracy (hard decision) for my training and validation data and ended up at around 84% correct predicitons after 30 epochs. When I tested the model on the test data that I set aside, I even got better results and an accuracy of **89%**.
* **Treating Silence** When you work with real-life data you always need to make sure to prepare it correctly. Background noises are sometimes desired because they make the model more robust. However, if the speaker is silent for a few seconds (even an entire sequence) it is not helpful in training. There is no information to be gained about silence. Thus, I pre-processed all of the audio files to remove silence using *sox*. After experimenting with parameters, I decide to delete everything with at least 1 second of silence and resume after 0.25 seconds of noise. Silence would be defined as *less than 1% of maximal volume*. In order to correctly predict the language of an audio file using my trained model, it would also be helpful to pre-process that file. I tested with a vocal recording of Freddie Mercury on Bohemian Rhapsody that was completely silent for more than half of the recording. Thus, my model could only predict the language 54%. After removing the silence, it was over 90% sure that Freddie Mercury was singing English.
* **Fun Side Note** Apparently the song "Yesterday" by The Beatles is 47% Hindi, who could have thought that? Of course, I did not train the model with music in the background so the sound of instruments is unknown to my neural network which makes it hard to classify it correctly. In order to correctly predict the language even with different sounds in the background, I would have to train the NN with much, much more data, even with worse quality, or as I explained music.


