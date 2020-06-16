import json
import tensorflow as tf
import numpy as np
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

with open("sarcasm.json", 'r') as f:
    datastore = json.load(f)

    sentences = []
    labels = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

training_sentences = sentences[0:training_size]


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

model = keras.models.load_model('NewsTitle-sarcasmDetection.h5')

while True:

    print('Enter your title:')
    inputvalue = input()
    inputlist = []
    inputlist.append(inputvalue)
    sequences = tokenizer.texts_to_sequences(inputlist)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    result = model.predict(padded)
    if result > 0.5: 
        print('this title is sarcastic') 
    else: 
        print('this title is not sarcastic')
    inputlist.clear()