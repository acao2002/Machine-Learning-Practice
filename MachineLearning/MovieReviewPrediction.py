import tensorflow as tf
from tensorflow import keras 

import numpy as np

model = keras.models.load_model("model.h5")
model.summary()

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=88000)

word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def review_encode(s):
    encoded = [1]

    for word in s: 
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded 
print((test_data[250]))
predict = model.predict([test_data[250]])
print(len(predict))

with open("animereview.txt", encoding="utf-8") as f:
    line = f.read()
    nline = line.replace(",", " ").replace(":", " ").replace(".", " ").replace(";", " ").replace("(", " ").replace(")", " ").replace("\"", " ").replace("-", " ")
    encode = review_encode(nline)
    encode = keras.preprocessing.sequence.pad_sequences([encode],
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
    animepredict = model.predict(encode[0])
    print(animepredict)   
    print(encode)
 
       