# Spark configuration and packages specification. The dependencies defined in
# this file will be automatically provisioned for each run that uses Spark.

import sys
import os
import pickle

from azureml.logging import get_azureml_logger
from azureml.dataprep import package

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.callbacks import Callback

logger = get_azureml_logger()


class AzureMlDataRecording(Callback):

    def on_epoch_end(self, epoch, logs=None):
        global logger

        logger.log('Training loss', float(logs.get('loss')))
        logger.log('Training accuracy', float(logs.get('acc')))


##
## STEP 1: Perform final prep work for training
##

# We did some prep work earlier that we use here.
# By running the data preparation packages we clean up the training and validation data.
train_data = package.run('training_prepared.dprep', dataflow_idx=0)
validation_data = package.run('validation_prepared.dprep', dataflow_idx=0)

# Fix up the intent column so it is categorical
train_data['intent'] = train_data['intent'].astype('category')
validation_data['intent'] = validation_data['intent'].astype('category')
validation_data['intent'].cat.categories = train_data['intent'].cat.categories.tolist()

# The neural network doesn't understand text at all.
# Therefor we convert the input text to vector representations using a tokenizer.
tokenizer = Tokenizer(num_words=2500)
tokenizer.fit_on_texts(train_data['text'].values)

# The training features and labels are made compatible with the neural network here.
# Notice the use of the tokenizer and a smart trick from pandas to convert the data.
features = tokenizer.texts_to_sequences(train_data['text'].values)
labels = to_categorical(train_data['intent'].astype('category').cat.codes)

# The validation features and labels are made compatible with the neural network here.
# Notice the use of the tokenizer and a smart trick from pandas to convert the data.
validation_features = tokenizer.texts_to_sequences(validation_data['text'].values)
validation_labels = to_categorical(validation_data['intent'].astype('category').cat.codes)

# We're reasoning over time (essentially each word in the sentence is a time-step).
# Let's calculate the maximum length of a sentence here so that all sentences can be made equal with padding.
sequence_length = max([len(x) for x in features])

# Apply padding, so the sequences are all of the same length.
features = pad_sequences(features, sequence_length)
validation_features = pad_sequences(validation_features, sequence_length)

##
## STEP 2: Build the model
##

model = Sequential()

# The first part of the model is an embedding layer that produces are more managable featureset.
# We're going to record 128 features per word in a sentence.
model.add(Embedding(len(tokenizer.word_index.keys()), 128, input_length=sequence_length))

# The second part of the neural network is a recurrent layer.
# This part of the network reasons over time and essentially learns the structure of a sentence.
model.add(LSTM(64, input_shape=(sequence_length, len(tokenizer.word_index.keys()))))

# The last part is a classic feed forward network.
# This part of the neural network learns how to put signatures of sentences into a specific category.
model.add(Dense(32, activation='relu'))
model.add(Dense(labels.shape[1], activation='softmax'))

# Print a summary for fun and giggles.
# This network has 1M+ parameters to train.
model.summary()

# Since this is a classification problem we need the categorical cross-entropy loss function.
# We optimize with stogastic gradient descent.
# To measure performance we record the accuracy.
model.compile('sgd', loss='categorical_crossentropy', metrics=['accuracy'])

##
## STEP 3: Train the model
##

# Train the model on the input data.
# For demo purposes I kept the amount of epochs low. No need to blow up hardware.
model.fit(features, labels, epochs=5, batch_size=32, callbacks=[AzureMlDataRecording()])

# Validate the model using the validation set.
# This gives me the metric that I need to see whether we're overfitting the model.
val_loss, val_acc = model.evaluate(validation_features, validation_labels)

logger.log('Validation accuracy', val_acc)

##
## STEP 4: Store the model on disk
##
os.makedirs('./outputs', exist_ok=True)

model.save('./outputs/model.h5')

# The tokenizer needs to be stored so we produce the right format input data for the neural network.
# Input text that needs to be classified is tokenized using this tokenizer.
with open('./outputs/tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

# The metadata file contains additional information that is necessary to preprocess input for predicition.
# Right now it only contains the value for the sequence length. But you could store more if you need.
with open('./outputs/metadata.pkl', 'wb') as metadata_file:
    metadata = {
        'sequence_length': sequence_length,
        'intents': train_data['intent'].astype('category').cat.categories.values
    }

    pickle.dump(metadata, metadata_file)
