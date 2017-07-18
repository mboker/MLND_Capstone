import tensorflow as tf
import pandas as pd
from collections import Counter
import math
import numpy as np


def get_data_and_vocab(file_path):
    data = pd.read_csv(file_path, header=0)
    headlines = data['Combined'][:].tolist()
    vocab = set()
    for headline in headlines:
        vocab.update(headline.split())
    return data, vocab


def get_data_splits(split_frac, data):
    labels = data['Label'][:].tolist()
    headline_strings = data['Combined'][:].tolist()
    split_idx = int(split_frac*len(labels))
    train_x, train_y = headline_strings[:split_idx], labels[:split_idx]
    test_x, test_y = headline_strings[split_idx:], labels[split_idx:]
    split_idx = int(split_frac*len(train_x))
    val_x, val_y = train_x[split_idx:], train_y[split_idx:]
    train_x, train_y = train_x[:split_idx], train_y[:split_idx]
    return train_x, val_x, test_x, train_y, val_y, test_y


def create_lists_and_filter(train_x, val_x, test_x):
    train_x = [[word for word in words.split()] for words in train_x]
    train_word_counter = Counter()
    for words in train_x:
        train_word_counter.update(words)
    train_x = [[word for word in words if train_word_counter[word] >= 5] for words in train_x]
    val_x = [[word for word in words if train_word_counter[word] >= 5] for words in val_x]
    test_x = [[word for word in words if train_word_counter[word] >= 5] for words in test_x]

    return train_x, val_x, test_x, train_word_counter


def get_batches_and_pad(headline_strings, labels, batch_size, length):
    num_batches = math.ceil(len(headline_strings) // batch_size)
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