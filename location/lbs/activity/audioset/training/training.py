#!/usr/bin/env python
# coding: utf-8

# In[ ]:


cd ../dataprocessing


# In[ ]:


# import audio_processing_test as apt
import audio_processing as ap

import functools
import os
import sys
from absl import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.utils import np_utils
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[ ]:


print(os.path.dirname(os.path.realpath('__file__')))


# In[ ]:


debug = False
logging.set_verbosity(logging.INFO)


# In[ ]:


src_dir = 'example_src_dir'
dest_dir = 'example_dest_dir'
# filenames should adhere to the following order
# [dataset, validation set, test set]
filenames = ['test_set']
labels = ['Gunshot, gunfire']
features_to_extract = ['mfcc']


# In[ ]:


def get_dataframes():
    length = len(filenames)
    if length == 3:
        dataset_df = ap.output_df(src_dir, dest_dir, filenames[0], labels, features_to_extract)
        evaluation_df = ap.output_df(src_dir, dest_dir, filenames[1], labels, features_to_extract)
        validation_df = ap.output_df(src_dir, dest_dir, filenames[2], labels, features_to_extract)
        dfs = [dataset_df, evaluation_df, validation_df]
    elif length == 2:
        dataset_df = ap.output_df(src_dir, dest_dir, filenames[0], labels, features_to_extract)
        evaluation_df = ap.output_df(src_dir, dest_dir, filenames[1], labels, features_to_extract)
        dfs = [dataset_df, evaluation_df]
    elif length == 1:
        dataset_df = ap.output_df(src_dir, dest_dir, filenames[0], labels, features_to_extract)
        dfs = [dataset_df]
    else:
        raise ValueError('You must have at least one dataset csv and testing data csv')
    return dfs


# In[ ]:


dfs = get_dataframes()


# In[ ]:


dataset_df = dfs[0]
dataset_df.head()


# Convert features and classification labels into numpy arrays

# In[ ]:


X = np.array(dataset_df.mfcc.tolist(), dtype=object)
y = np.array(dataset_df.label.tolist())


# In[ ]:


def get_data_for_model(dfs, ratio):
    length = len(dfs)
    if length == 3:
        train_x = np.array(dfs[0].mfcc.tolist(), dtype=object)
        train_y = np.array(dfs[0].label.tolist())
        test_x = np.array(dfs[1].mfcc.tolist(), dtype=object)
        test_y = np.array(dfs[1].label.tolist())
        val_x = np.array(dfs[2].mfcc.tolist(), dtype=object)
        val_y = np.array(dfs[2].label.tolist())
        return train_x, train_y, val_x, val_y, test_x, test_y
    elif length == 2:
        train_x = np.array(dfs[0].mfcc.tolist(), dtype=object)
        train_y = np.array(dfs[0].label.tolist())
        test_x = np.array(dfs[1].mfcc.tolist(), dtype=object)
        test_y = np.array(dfs[1].label.tolist())
        return train_x, train_y, test_x, test_y
    elif length == 1:
        X = np.array(dataset_df.mfcc.tolist(), dtype=object)
        y = np.array(dataset_df.label.tolist())
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state = 42)
        return train_x, train_y, test_x, test_y


# # Split the dataset

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state = 42)


# In[ ]:


# trying to fix bug:
# ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float).
# It worked!!!
from keras import backend as K
x_train = K.cast_to_floatx(x_train)
y_train = K.cast_to_floatx(y_train)
x_test = K.cast_to_floatx(x_test)
y_test = K.cast_to_floatx(y_test)


# In[ ]:


data = (x_train, x_test, y_train, y_test)


# In[ ]:


def visualize_training(history, filename):
    history_dict = history.history
    history_dict.keys()
    acc = history_dict['accuracy']
#     val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
#     val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
    
    # "bo" is for "blue dot"
    ax1.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
#     ax1.plot(epochs, val_loss, 'b', label='Validation loss')
    ax1.set_title('Training and validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(epochs, acc, 'bo', label='Training acc')
#     ax2.plot(epochs, val_acc, 'b', label='Validation acc')
    ax2.set_title('Training and validation accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='lower right')
    
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


# In[ ]:


def model_config1(activation, optimizer, metrics):
    model = keras.Sequential([
        keras.Input(shape=(20,)),
        keras.layers.Dense(20, activation=activation),
        keras.layers.Dense(1, activation=activation)
    ])
    model.compile(optimizer=optimizer,
             loss=tf.keras.losses.BinaryCrossentropy(),
             metrics=metrics)
    return model


# In[ ]:


def model_train1(model, data, epochs):
    x_train, x_test, y_train, y_test = data
    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        verbose=1,
                        use_multiprocessing=False
    )
    return history


# In[ ]:


def print_performance(history):
    loss = history.history.get('loss')
    accuracy = history.history.get('accuracy')
    tp = history.history.get('tp')
    fp = history.history.get('fp')
    tn = history.history.get('tn')
    fn = history.history.get('fn')
    print('Loss: {}'.format(loss))
    print('accuracy: {}'.format(accuracy))
    print('True Positives: {}'.format(tp))
    print('False Positives: {}'.format(fp))
    print('True Negatives: {}'.format(tn))
    print('False Negatives: {}'.format(fn))
    prec = []
    rec = []
    f1 = []
    for tp1, fp1 in list(zip(tp, fp)):
        prec.append(precision(tp1, fp1))
    for tp1, fn1 in list(zip(tp, fn)):
        rec.append(recall(tp1, fn1))
    for rec1, prec1 in list(zip(rec, prec)):
        f1.append(f1score(rec1, prec1))
        print('Recall: {}'.format(rec))
    print('Precision: {}'.format(prec))
    print('F1-Score: {}'.format(f1))


# In[ ]:


def precision(tp, fp):
    sum = tp + fp
    if sum == 0:
        return 0
    return tp / sum


# In[ ]:


def recall(tp, fn):
    sum = tp + fn
    if sum == 0:
        return 0
    return tp / sum


# In[ ]:


def f1score(recall, precision):
    sum = recall + precision
    if sum == 0:
        return 0
    return 2 * recall * precision / sum


# In[ ]:


path = os.path.join(dest_dir, 'results1')
metrics_list = [['accuracy'], 
                [tf.keras.metrics.TruePositives(name='tp')], 
                [tf.keras.metrics.TrueNegatives()], 
                [tf.keras.metrics.FalseNegatives()], 
                [tf.keras.metrics.FalsePositives()]
               ]
metrics = [
    'accuracy',
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
]
model = model_config1('softmax', 'adam', metrics)
history = model_train1(model, data, 20)
visualize_training(history, path)
print_performance(history)


# In[ ]:




