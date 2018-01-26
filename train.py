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


logger = get_azureml_logger()

train_data = package.run('training_prepared.dprep', dataflow_idx=0)
validation_data = package.run('validation_prepared.dprep', dataflow_idx=0)

tokenizer = Tokenizer(num_words=2500)

tokenizer.fit_on_texts(train_data['text'].values)

features = tokenizer.texts_to_sequences(train_data['text'].values)
labels = to_categorical(train_data['intent'].astype('category').cat.codes)

validation_features = tokenizer.texts_to_sequences(validation_data['text'].values)
validation_labels = to_categorical(validation_data['intent'].astype('category').cat.codes)

sequence_length = max([len(x) for x in features])

features = pad_sequences(features, sequence_length)
validation_features = pad_sequences(validation_features, sequence_length)

model = Sequential()

model.add(Embedding(len(tokenizer.word_index.keys()), 128, input_length=sequence_length))
model.add(LSTM(64, input_shape=(sequence_length, len(tokenizer.word_index.keys()))))
model.add(Dense(32, activation='relu'))
model.add(Dense(labels.shape[1], activation='softmax'))

model.summary()

model.compile('sgd', loss='categorical_crossentropy', metrics=['accuracy'])

train_history = model.fit(features, labels, epochs=5, batch_size=32)

logger.log('Training accuracy', train_history.history['acc'])

validation_results = model.evaluate(validation_features, validation_labels)

logger.log('Validation accuracy', validation_results)

os.makedirs('./outputs', exist_ok=True)

model.save('./outputs/model.h5')

with open('./outputs/tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)