import pickle
import numpy as np

from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.

model = None
tokenizer = None
metadata = None


def init():
    # Model and tokenizer are stored globally.
    global model, tokenizer, metadata

    # Load the keras model from disk
    model = load_model('model.h5')

    # Load the tokenizer from disk
    with open('tokenizer.pkl', 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    with open('metadata.pkl', 'rb') as metadata_file:
        metadata = pickle.load(metadata_file)


def run(text):
    import json

    input_sequence = tokenizer.texts_to_sequences([text])
    input_sequence = pad_sequences(input_sequence, metadata['sequence_length'])

    prediction = model.predict(input_sequence)

    intent_mapping = metadata['intents']

    return json.dumps({ 'intent': intent_mapping[int(np.argmax(prediction))] })


def generate_api_schema():
    import os
    print("create schema")
    sample_input = "sample data text"
    inputs = {"text": SampleDefinition(DataTypes.STANDARD, sample_input)}
    os.makedirs('outputs', exist_ok=True)
    print(generate_schema(inputs=inputs, filepath="outputs/schema.json", run_func=run))


# Implement test code to run in IDE or Azure ML Workbench
if __name__ == '__main__':
    # Import the logger only for Workbench runs
    from azureml.logging import get_azureml_logger

    logger = get_azureml_logger()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', help='Generate Schema')
    args = parser.parse_args()

    if args.generate:
        generate_api_schema()

    init()

    input = "Play a beatles song"
    result = run(input)

    logger.log("Result", result)
