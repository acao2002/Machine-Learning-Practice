import tensorflow as tf
from tensorflow import keras

import numpy as np

imdb = keras.datasets.imdb
word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3



def review_encode(s):
    encoded = [1]

    for word in s: 
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded 

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

model = keras.models.load_model("movieprediction.h5")

with open("animereview.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(":", "").replace(".", "").replace(";", "").replace("(", "").replace(")", "").replace("\"", "")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode],
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
        
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict)
       