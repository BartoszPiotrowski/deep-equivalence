#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import sys
sys.path.append('')
from utils import dataset as data
# ~DONE control all shapes
# DONE check how data set is balanced -> 42% are positives, good enough
# TODO put all parameters as args to parse
# TODO commas v no commas
# TODO grid search
# TODO saving the network
# TODO check which formulae are problematic
# TODO solve some examples by hand

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
            self.tokens_ids_1 = tf.placeholder(
                tf.int32, [None, None], name="tokens_ids_1")
            self.formulae_lens_1 = tf.placeholder(
                tf.int32, [None], name="formulae_lens_1")
            self.tokens_ids_2 = tf.placeholder(
                tf.int32, [None, None], name="tokens_ids_2")
            self.formulae_lens_2 = tf.placeholder(
                tf.int32, [None], name="formulae_lens_2")
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

            #print('state_fwd_1 shape :', state_fwd_1.get_shape())
            print('state_fwd_1 :', state_fwd_1)
            state_1 = tf.concat([state_fwd_1, state_bwd_1], axis=-1)
            state_2 = tf.concat([state_fwd_2, state_bwd_2], axis=-1)
            print('state_1 shape :', state_1.get_shape())
            print('state_2 shape :', state_2.get_shape())
            state = tf.concat([state_1, state_2], axis=-1)
            print('state shape :', state.get_shape())
            layers = [state]
            for _ in range(args.num_dense_layers):
                layers.append(
                    tf.layers.dense(layers[-1],
                                    args.dense_layer_units,
                                    activation=tf.nn.relu)
                )
            logits = tf.layers.dense(layers[-1], self.LABELS, name='logits')
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

            # Saver
            tf.add_to_collection('end_points/tokens_ids_1',
                                 self.tokens_ids_1)
            tf.add_to_collection('end_points/formulae_lens_1',
                                 self.formulae_lens_1)
            tf.add_to_collection('end_points/tokens_ids_2',
                                 self.tokens_ids_2)
            tf.add_to_collection('end_points/formulae_lens_2',
                                 self.formulae_lens_2)
            tf.add_to_collection('end_points/predictions',
                                 self.predictions)

            self.saver = tf.train.Saver()

    def save(self, path):
        return self.saver.save(self.session, path)

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

class NetworkPredict:
    def __init__(self, threads=1, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(
            graph=graph,
            config=tf.ConfigProto(
                inter_op_parallelism_threads=threads,
                intra_op_parallelism_threads=threads))

    def load(self, path):
        # Load the metagraph
        with self.session.graph.as_default():
            self.saver = tf.train.import_meta_graph(path + '.meta')

            # Attach the end points
            self.tokens_ids_1 = tf.get_collection(
                'end_points/tokens_ids_1')[0]
            self.formulae_lens_1 = tf.get_collection(
                'end_points/formulae_lens_1')[0]
            self.tokens_ids_2 = tf.get_collection(
                'end_points/tokens_ids_2')[0]
            self.formulae_lens_2 = tf.get_collection(
                'end_points/formulae_lens_2')[0]
            self.predictions = tf.get_collection(
                'end_points/predictions')[0]

        # Load the graph weights
        self.saver.restore(self.session, path)

    def predict(self, dataset_name, dataset):
        return self.session.run(self.predictions,
                         {self.formulae_lens_1: formulae_lens_1,
                          self.tokens_ids_1: tokens_ids_1,
                          self.formulae_lens_2: formulae_lens_2,
                          self.tokens_ids_2: tokens_ids_2})

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
        "--vocab",
        default='data/vocab.txt',
        type=str,
        help="Path to a vocabulary file.")
    parser.add_argument(
        "--train_set",
        default='data/split/equiv.train',
        type=str,
        help="Path to a training set.")
    parser.add_argument(
        "--valid_set",
        default='data/split/equiv.valid',
        type=str,
        help="Path to a validation (dev) set.")
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size.")
    parser.add_argument(
        "--epochs",
        default=16,
        type=int,
        help="Number of epochs.")
    parser.add_argument(
        "--embed_dim",
        default=16,
        type=int,
        help="Token embedding dimension.")
    parser.add_argument(
        "--rnn_cell_dim",
        default=16,
        type=int,
        help="RNN cell dimension.")
    parser.add_argument(
        "--num_dense_layers",
        default=2,
        type=int,
        help="Number of dense layers.")
    parser.add_argument(
        "--dense_layer_units",
        default=16,
        type=int,
        help="Number of units in each dense layer.")
    parser.add_argument(
        "--threads",
        default=2,
        type=int,
        help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create dir for logs
    if not os.path.exists("logs"):
        os.mkdir("logs")

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(
            ("{}={}".format(
                re.sub("(.)[^_]*_?", r"\1", key), value) \
                    for key, value in sorted(vars(args).items()))))

    # Load the data
    train = data.Dataset(args.train_set, args.vocab, shuffle_batches=True)
    dev = data.Dataset(args.valid_set, args.vocab, shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, train.num_tokens)

    # Train
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size)
        accuracy = network.evaluate("dev", dev, args.batch_size)
        print("Accuracy on dev set: {:.2f}".format(100 * accuracy))
    print("Training finished.")

    # Save
    model_path = 'saved_models/model-1-test'
    model_path = network.save(model_path)
    print(model_path)

    # Predict on test set
    network = NetworkPredict()
    network.load(model_path)