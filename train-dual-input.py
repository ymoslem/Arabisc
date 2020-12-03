import tensorflow
print(tensorflow.__version__)

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import *
from tensorflow.python.client import device_lib
import numpy as np
import sys
import os


# Activate GPUs
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"       # Change if you have 1 GPU

# Check devices visible to TensorFlow
print(device_lib.list_local_devices())


# Pre-process data
data_file = sys.argv[1]                       # Dataset file as an argument
data = open(data_file).read()

corpus = data.lower().split("\n")
print(corpus[0])


vocab_size = 100000
out_of_vocab = "<unk>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token=out_of_vocab)
tokenizer.fit_on_texts(corpus)


# Create input sequences using list of tokens
input_sequences = []            # words before the current word/label
input_sequences_reversed = []   # words after the current word/label
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
        n_gram_sequence_rev = token_list[i:]
        n_gram_sequence_rev.reverse()
        input_sequences_reversed.append(n_gram_sequence_rev)


# Pad sequences
max_sequence_len = 15
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
input_sequences_reversed = np.array(pad_sequences(input_sequences_reversed, maxlen=max_sequence_len, padding='pre'))


# Create predictors (words before and after) and label (current word)
predictors, label = input_sequences[:,:-1], input_sequences[:,-1]
predictors_rev, label_rev = input_sequences_reversed[:,:-1], input_sequences_reversed[:,-1]

# Test - label must be the same as label_rev, i.e. the current word
print(predictors[1], label[1])
print(predictors_rev[1], label_rev[1])


# Build the model - 2 inputs, 1 output

# Input
left_input = Input(shape=(max_sequence_len-1,), name="left_in")
right_input = Input(shape=(max_sequence_len-1,),  name="right_in")

# Embedding
left_features = Embedding(vocab_size, 256)(left_input)
right_features = Embedding(vocab_size, 256)(right_input)

# Bidirectional LSTM
left_features = Bidirectional(LSTM(512, return_sequences = True))(left_features)
right_features = Bidirectional(LSTM(512, return_sequences = True))(right_features)

# LSTM
left_features = LSTM(128)(left_features)
right_features = LSTM(128)(right_features)

# Merge all available features into a single large vector via concatenation
concat = concatenate([left_features, right_features], name='concatenate')

# Dense - output
pred = Dense(vocab_size, activation='softmax', name="pred")(concat)


# Instantiate an end-to-end model predicting the next word based on both the left and right inputs
model = Model(
    inputs = [left_input, right_input],
    outputs = pred
)


model.compile(
    optimizer = Adam(),
    loss = SparseCategoricalCrossentropy(),
    metrics = ['accuracy']
)

print(model.summary())


# Define callbacks
my_callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=3),
    ModelCheckpoint(filepath='model-dual-input/model.{epoch:02d}-{accuracy:.3f}-{val_accuracy:.3f}', verbose=1)
]


# Train the model
model.fit(
    {"left_in": predictors, "right_in": predictors_rev},
    {'pred': label},
    epochs=200,
    batch_size=128,
    validation_split=0.02,
    callbacks=my_callbacks
)
