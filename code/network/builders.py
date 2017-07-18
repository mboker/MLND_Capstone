import tensorflow as tf
import pandas as pd
from collections import Counter
import numpy as np
import random


# get_data_and_vocab opens the csv file containing our combined and tokenized headline strings
# it then creates a set of all of the word ids contained in the set
# it returns the data, along with the vocab set
def get_data_and_vocab(file_path):
    data = pd.read_csv(file_path, header=0)
    headlines = data['Combined'][:].tolist()
    vocab = set()
    for headline in headlines:
        vocab.update(headline.split())
    return data, vocab


# Splits the data into training, test, and validation sets, split_frac
# is used to split the training set from data, and the remaining points are
# split evenly between validation and test
def get_data_splits(split_frac, data):
    labels = data['Label'][:].tolist()
    headline_strings = data['Combined'][:].tolist()
    split_idx = int(split_frac*len(labels))
    train_x, train_y = headline_strings[:split_idx], labels[:split_idx]
    test_x, test_y = headline_strings[split_idx:], labels[split_idx:]
    split_idx = int(0.5*len(test_x))
    val_x, val_y = test_x[:split_idx], test_y[:split_idx]
    test_x, test_y = test_x[split_idx:], test_y[split_idx:]
    return train_x, val_x, test_x, train_y, val_y, test_y


# subsample uses the subsampling formula from Mikolov, et al to probablistically
# discard words.  As it is used in the list comprehensions in create_lists_and_filter,
# if it returns False, the word being subsampled is discarded, otherwise it is included
def subsample(freq, thresh):
    p = (freq-thresh)/freq - (thresh/freq)**(0.5)
    if random.random() <= p:
        return False
    return True


# this method takes the datasets and turns their headline strings into lists of word ids,
# discarding any words that appear fewer than 5 times throughout the data set, and
# subsampling all other words
def create_lists_and_filter(train_x, val_x, test_x, thresh):
    train_x = [[word for word in words.split()] for words in train_x]
    train_word_counter = Counter()
    for words in train_x:
        train_word_counter.update(words)
    total_count = sum(train_word_counter.values())
    train_x = [[word for word in words if train_word_counter[word] >= 5 and
                subsample(train_word_counter[word]/total_count,thresh)] for words in train_x]
    val_x = [[word for word in words if train_word_counter[word] >= 5] for words in val_x]
    test_x = [[word for word in words if train_word_counter[word] >= 5] for words in test_x]

    return train_x, val_x, test_x, train_word_counter


# get_batches_and_pad takes in sets of headlines_strings and labels, of equal size,
# and splits them into batches, each of size batch_size.  Any remaining data points
# that do not fit into a full batch are discarded.  the headline strings are also
# padded or truncated to be exactly length words long
def get_batches_and_pad(headline_strings, labels, batch_size, length):
    num_batches = len(headline_strings) // batch_size
    batches = []
    for batch_num in range(num_batches):
        batch_headlines = headline_strings[batch_num*batch_size: min(len(headline_strings), (batch_num+1)*batch_size)]
        padded_and_trimmed_headlines = []
        for headline_string in batch_headlines:
            if len(headline_string) >= length:
                padded_and_trimmed_headlines.append(headline_string[:length])
            else:
                padded_and_trimmed_headlines.append([0 for i in range(length-len(headline_string))] + list(map(int, headline_string)))
        batch_labels = labels[batch_num*batch_size: min(len(labels), (batch_num+1)*batch_size)]
        batches.append({'headlines': padded_and_trimmed_headlines, 'labels': np.array(batch_labels)})
    return np.array(batches)


def build_inputs_and_targets():
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')

    return inputs, targets


def build_lstm_cell(rnn_size, batch_size, output_keep_prob):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    dropout = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=output_keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([dropout])

    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, name='initial_state')

    return cell, initial_state


def build_embedding_layer(input_data, vocab_size, embedding_dim):
    embedding = tf.Variable(tf.random_uniform((vocab_size, embedding_dim), -1, 1))
    return tf.nn.embedding_lookup(embedding, input_data)


def build_rnn(cell, inputs):
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, 'final_state')
    return outputs, final_state
