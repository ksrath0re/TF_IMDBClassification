#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras

import numpy as np
import itertools

print(tf.__version__)


# In[4]:


imdb = keras. datasets. imdb
(training_data, training_label), (test_data, test_label) = imdb.load_data(num_words=10000)
#The IMDB dataset comes packaged with TensorFlow. It has already been preprocessed such that the reviews 
#(sequences of words) have been converted to sequences of integers, where each integer represents a 
#specific word in a dictionary.


# In[5]:


print("Training entries : {}, labels: {}".format(len(training_data), len(training_data)))


# In[6]:


#Let's test how review looks like in this processed data
print(training_data[0])


# In[7]:


print(len(training_data[0]), len(training_data[20]))


# In[8]:


temp = imdb.get_word_index()
for key,value in list(temp.items())[0:1000]:
    if temp[key] < 2000:
        print(key, value)


# In[9]:


word_index = imdb.get_word_index()
#let's reserve first 3 indices for dealing with NL stuff

word_index = {key:(value+3) for key,value in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNKNOWN>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for key,value in word_index.items()])

def decode_text_from_intarr(integerText):
    return ' '.join(reverse_word_index.get(i, '?') for i in integerText)


# In[10]:


decode_text_from_intarr(training_data[1])


# In[11]:


training_data = keras.preprocessing.sequence.pad_sequences(training_data, value=word_index["<PAD>"],padding='post',maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"],padding='post',maxlen=256)


# In[12]:


print(len(training_data[0]), len(training_data[0]))


# In[13]:


vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

print(model.summary())


# In[14]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


# In[15]:


x_val = training_data[:10000]
partial_x_train = training_data[10000:]

y_val = training_label[:10000]
partial_y_train = training_label[10000:]


# In[16]:


training_details = model.fit(partial_x_train, partial_y_train,epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)


# In[17]:


results = model.evaluate(test_data, test_label)
print(results)


# In[18]:


history_dic = training_details.history
print(history_dic.keys())


# In[22]:


import matplotlib.pyplot as p
accuracy = history_dic['acc']
val_acc = history_dic['val_acc']
loss = history_dic['loss']
val_loss = history_dic['val_loss']

epochs = range(1,len(accuracy)+1)

# "bo" is for "blue dot"
p.plot(epochs, loss, 'bo', label='Training Loss')
#b is for solid blue line
p.plot(epochs, val_loss, 'b', label='Validation Loss')
p.title('Training and Validation Loss')
p.xlabel('Epochs')
p.ylabel('Loss')
p.legend()
p.show()


# In[24]:


p.clf()

p.plot(epochs, accuracy, 'bo', label='Training Accuracy')
#b is for solid blue line
p.plot(epochs, val_acc, 'b', label='Validation Accuracy')
p.title('Training and Validation Accuracy')
p.xlabel('Epochs')
p.ylabel('Accuracy')
p.legend()
p.show()

