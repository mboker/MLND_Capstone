import tensorflow as tf
import builders
import pandas as pd
import numpy as np

file_path = '../../data/tokenized_headlines.csv'
#string creation and batch hyperparameters
split_frac = 0.9
batch_size = 179
length = 400
#network hyperparameters
num_epochs = 200
rnn_size = 256
embed_dim = 150
learning_rate = 0.0001
dropout_keep_rate = 0.3


data, vocab = builders.get_data_and_vocab(file_path)
train_x, val_x, test_x, train_y, val_y, test_y = builders.get_data_splits(split_frac, data)
train_x, val_x, test_x = builders.create_lists_and_filter(train_x, val_x, test_x)
train_batches = builders.get_batches_and_pad(train_x, train_y, batch_size, length)
val_batches = builders.get_batches_and_pad(val_x, val_y, batch_size, length)

vocab_size = len(vocab)

#build network
train_graph = tf.Graph()
with train_graph.as_default():
    inputs, targets = builders.build_inputs_and_targets()
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    input_shape = tf.shape(inputs)
    cell, initial_state = builders.build_lstm_cell(rnn_size, batch_size, keep_prob)
    embedding_layer = builders.build_embedding_layer(inputs, vocab_size, embed_dim)
    rnn, final_state = builders.build_rnn(cell, embedding_layer)
    predictions = tf.contrib.layers.fully_connected(inputs=rnn[:,-1],
                                               num_outputs=1,
                                               activation_fn=tf.sigmoid,
                                               weights_initializer=tf.truncated_normal_initializer(),
                                               biases_initializer=tf.zeros_initializer(),
                                               trainable=True)
    cost = tf.losses.mean_squared_error(targets, predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    saver = tf.train.Saver()
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), targets)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(num_epochs):
        state = sess.run(initial_state)

        for index, batch in enumerate(train_batches):
            feed = {inputs: batch['headlines'],
                    targets: batch['labels'][:, None],
                    keep_prob: dropout_keep_rate,
                    initial_state: state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

            if iteration%5==0:
                print('Epoch: {}/{}'.format(e, num_epochs),
                      'Iteration: {}'.format(iteration),
                      'Train Loss: {:.3f}'.format(loss))

            # VALIDATE
            if iteration % 25 == 0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for val_index, val_batch in enumerate(val_batches):
                    feed = {inputs: val_batch['headlines'],
                            targets: val_batch['labels'][:, None],
                            keep_prob: dropout_keep_rate,
                            initial_state: state}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Validation Accuracy: {:.3f}".format(np.mean(val_acc)))
            iteration += 1
        saver.save
