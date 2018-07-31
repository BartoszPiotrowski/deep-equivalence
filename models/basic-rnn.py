#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import sys
sys.path.append('')
from utils import dataset as data
# TODO control all shapes
# TODO check how data set is balanced
# TODO check which formulae are problematic
# TODO grid search

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(
            graph=graph,
            config=tf.ConfigProto(
                inter_op_parallelism_threads=threads,
                intra_op_parallelism_threads=threads))

    def construct(self, args, num_tokens):
        self.LABELS = 2
        with self.session.graph.as_default():
            # Inputs
            self.formulae_lens_1 = tf.placeholder(
                tf.int32, [None], name="formulae_lens_1")
            self.formulae_lens_2 = tf.placeholder(
                tf.int32, [None], name="formulae_lens_2")
            self.tokens_ids_1 = tf.placeholder(
                tf.int32, [None, None], name="tokens_ids_1")
            self.tokens_ids_2 = tf.placeholder(
                tf.int32, [None, None], name="tokens_ids_2")
            self.labels = tf.placeholder(tf.int32, [None], name="labels")

            # Token embeddings
            token_embeddings = tf.get_variable(
                "token_embeddings",
                shape=[num_tokens, args.embed_dim],
                dtype=tf.float32)
            print('token_embeddings shape :', token_embeddings.get_shape())
            inputs_1 = tf.nn.embedding_lookup(token_embeddings, self.tokens_ids_1)
            inputs_2 = tf.nn.embedding_lookup(token_embeddings, self.tokens_ids_2)
            print('inputs_1 shape :', inputs_1.get_shape())
            print('inputs_2 shape :', inputs_2.get_shape())

            # Computation
            # rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
            rnn_cell = tf.nn.rnn_cell.GRUCell

            with tf.variable_scope('bi_rnn_1'):
                _, (state_fwd_1, state_bwd_1) = \
                        tf.nn.bidirectional_dynamic_rnn(
                            rnn_cell(args.rnn_cell_dim),
                            rnn_cell(args.rnn_cell_dim),
                            inputs_1,
                            sequence_length=self.formulae_lens_1,
                            dtype=tf.float32)
            with tf.variable_scope('bi_rnn_2'):
                _, (state_fwd_2, state_bwd_2) = \
                        tf.nn.bidirectional_dynamic_rnn(
                            rnn_cell(args.rnn_cell_dim),
                            rnn_cell(args.rnn_cell_dim),
                            inputs_2,
                            sequence_length=self.formulae_lens_2,
                            dtype=tf.float32)
            # TODO are axis OK?
            #print('state_fwd_1 shape :', state_fwd_1.get_shape())
            print('state_fwd_1 :', state_fwd_1)
            state_1 = tf.concat([state_fwd_1, state_bwd_1], axis=-1)
            state_2 = tf.concat([state_fwd_2, state_bwd_2], axis=-1)
            print('state_1 shape :', state_1.get_shape())
            print('state_2 shape :', state_2.get_shape())
            state = tf.concat([state_1, state_2], axis=-1)
            print('state shape :', state.get_shape())
            layer_1 = tf.layers.dense(state,
                                      args.dense_layer_dim,
                                      activation=tf.nn.relu)
            logits = tf.layers.dense(layer_1, self.LABELS, name='logits')
            self.predictions = tf.argmax(logits, axis=1, name='predictions')
            #predictions_shape = tf.shape(self.predictions)
            print('self.predictions shape: ', self.predictions.get_shape())

            # Training
            #labels_shape = tf.shape(self.labels)
            print('self.labels shape: ', self.labels.get_shape())
            #logits_shape = tf.shape(logits)
            print('logits shape: ', logits.get_shape())
            loss = tf.losses.sparse_softmax_cross_entropy(
                self.labels, logits)
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(
                loss, global_step=global_step, name="training")

            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(
                self.labels, self.predictions)
            self.current_loss, self.update_loss = tf.metrics.mean(loss)
            self.reset_metrics = tf.variables_initializer(
                tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(
                args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), \
                    tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [
                    tf.contrib.summary.scalar(
                        "train/loss",
                        self.update_loss),
                    tf.contrib.summary.scalar(
                        "train/accuracy",
                        self.update_accuracy)]
            with summary_writer.as_default(), \
                    tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [
                        tf.contrib.summary.scalar(
                            dataset + "/loss",
                            self.current_loss),
                        tf.contrib.summary.scalar(
                            dataset + "/accuracy",
                            self.current_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(
                    session=self.session, graph=self.session.graph)

    def train_epoch(self, train, batch_size):
        while not train.epoch_finished():
            tokens_ids_1, formulae_lens_1, \
            tokens_ids_2, formulae_lens_2, \
                labels = train.next_batch(batch_size)
            self.session.run(self.reset_metrics)
            self.session.run([self.training, self.summaries["train"]],
                             {self.formulae_lens_1: formulae_lens_1,
                              self.tokens_ids_1: tokens_ids_1,
                              self.formulae_lens_2: formulae_lens_2,
                              self.tokens_ids_2: tokens_ids_2,
                              self.labels: labels})

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            tokens_ids_1, formulae_lens_1, \
            tokens_ids_2, formulae_lens_2, \
                labels = dataset.next_batch(batch_size)
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.formulae_lens_1: formulae_lens_1,
                              self.tokens_ids_1: tokens_ids_1,
                              self.formulae_lens_2: formulae_lens_2,
                              self.tokens_ids_2: tokens_ids_2,
                              self.labels: labels})
        return self.session.run(
            [self.current_accuracy, self.summaries[dataset_name]])[0]


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        default=10,
        type=int,
        help="Batch size.")
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="Number of epochs.")
    parser.add_argument(
        "--rnn_cell",
        default="LSTM",
        type=str,
        help="RNN cell type.")
    parser.add_argument(
        "--dense_layer_dim",
        default=32,
        type=int,
        help="number of units in dense layer.")
    parser.add_argument(
        "--rnn_cell_dim",
        default=32,
        type=int,
        help="RNN cell dimension.")
    parser.add_argument(
        "--threads",
        default=1,
        type=int,
        help="Maximum number of threads to use.")
    parser.add_argument(
        "--embed_dim",
        default=32,
        type=int,
        help="Token embedding dimension.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(
            ("{}={}".format(
                re.sub(
                    "(.)[^_]*_?",
                    r"\1",
                    key),
                value) for key,
                value in sorted(
                vars(args).items()))))
    if not os.path.exists("logs"): # TODO to remove
        os.mkdir("logs")  # TF 1.6 will do this by itself

    # Load the data
    train = data.Dataset('data/split/equiv.train', 'data/vocab.txt')
    dev = data.Dataset('data/split/equiv.valid', 'data/vocab.txt',
                       shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    # TODO this '+ 1' shouldn't be there
    # network.construct(args, train.num_tokens)
    network.construct(args, train.num_tokens + 1)

    # Train
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size)
        accuracy = network.evaluate("dev", dev, args.batch_size)
        print("{:.2f}".format(100 * accuracy))
