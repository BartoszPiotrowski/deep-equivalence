#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import sys
from time import time
sys.path.append('')
from utils import dataset as data
# DONE check how data set is balanced -> 42% are positives, good enough
# TODO commas v no commas
# TODO grid search
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
            self.token_embeddings = tf.get_variable(
                "token_embeddings",
                shape=[num_tokens, args.embed_dim],
                dtype=tf.float32)
            inputs_1 = tf.nn.embedding_lookup(self.token_embeddings,
                                              self.tokens_ids_1)
            inputs_2 = tf.nn.embedding_lookup(self.token_embeddings,
                                              self.tokens_ids_2)
            #print('inputs_1 shape :', inputs_1.get_shape())
            #print('inputs_2 shape :', inputs_2.get_shape())

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
            #print('state_fwd_1 :', state_fwd_1)
            state_1 = tf.concat([state_fwd_1, state_bwd_1], axis=-1)
            state_2 = tf.concat([state_fwd_2, state_bwd_2], axis=-1)
            #print('state_1 shape :', state_1.get_shape())
            #print('state_2 shape :', state_2.get_shape())
            state = tf.concat([state_1, state_2], axis=-1)
            #print('state shape :', state.get_shape())
            layers = [state]
            for _ in range(args.num_dense_layers):
                layers.append(
                    tf.layers.dense(layers[-1],
                                    args.dense_layer_units,
                                    activation=tf.nn.relu)
                )
            self.logits = tf.layers.dense(layers[-1], self.LABELS, name='logits')
            self.logits = tf.nn.softmax(self.logits)
            self.logits_0 = self.logits[:0]
            self.predictions = tf.argmax(self.logits, axis=1, name='predictions')
            #predictions_shape = tf.shape(self.predictions)
            #print('self.predictions shape: ', self.predictions.get_shape())

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(
                self.labels, self.logits)
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
                self.summaries['train'] = [
                    tf.contrib.summary.scalar(
                        'train/loss',
                        self.update_loss),
                    tf.contrib.summary.scalar(
                        'train/accuracy',
                        self.update_accuracy)]
            with summary_writer.as_default(), \
                    tf.contrib.summary.always_record_summaries():
                self.summaries['valid'] = [
                    tf.contrib.summary.scalar(
                        'valid/loss',
                        self.current_loss),
                    tf.contrib.summary.scalar(
                        'valid/accuracy',
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
            tf.add_to_collection('end_points/logits',
                                 self.logits)

            self.saver = tf.train.Saver()

    def save(self, path):
        return self.saver.save(self.session, path)

    def train_epoch(self, train_set, batch_size):
        while not train_set.epoch_finished():
            tokens_ids_1, formulae_lens_1, \
            tokens_ids_2, formulae_lens_2, \
                labels = train_set.next_batch(batch_size)
            self.session.run(self.reset_metrics)
            self.session.run([self.training, self.summaries["train"]],
                             {self.formulae_lens_1: formulae_lens_1,
                              self.tokens_ids_1: tokens_ids_1,
                              self.formulae_lens_2: formulae_lens_2,
                              self.tokens_ids_2: tokens_ids_2,
                              self.labels: labels})

    def train_batch(self, batch):
        tokens_ids_1, formulae_lens_1, \
        tokens_ids_2, formulae_lens_2, \
                            labels = batch
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

    def embeddings(self):
        return self.session.run(self.token_embeddings)

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
        tokens_ids_1, formulae_lens_1, \
        tokens_ids_2, formulae_lens_2, \
            = dataset.test()
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
        default='data/vocab',
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
        help="Path to a validation set.")
    parser.add_argument(
        "--test_set",
        default='',
        type=str,
        help="Path to a testing set.")
    parser.add_argument(
        "--model_path",
        default='',
        type=str,
        help="Path where to save the trained model.")
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size.")
    parser.add_argument(
        "--epochs",
        default=8,
        type=int,
        help="Number of epochs.")
    parser.add_argument(
        "--embed_dim",
        default=8,
        type=int,
        help="Token embedding dimension.")
    parser.add_argument(
        "--rnn_cell_dim",
        default=8,
        type=int,
        help="RNN cell dimension.")
    parser.add_argument(
        "--num_dense_layers",
        default=2,
        type=int,
        help="Number of dense layers.")
    parser.add_argument(
        "--dense_layer_units",
        default=8,
        type=int,
        help="Number of units in each dense layer.")
    parser.add_argument(
        "--threads",
        default=1,
        type=int,
        help="Maximum number of threads to use.")
    parser.add_argument(
        "--logdir",
        default='',
        type=str,
        help="Logdir.")
    args = parser.parse_args()

    if args.model_path:
        logdir = args.model_path
    else:
        if not args.logdir:
            # Create dir for logs
            if not os.path.exists("logs"):
                os.mkdir("logs")

            # Create logdir name
            logdir = "logs/{}--{}--{}".format(
                os.path.basename(__file__),
                datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"),
                ",".join(
                    ("{}={}".format(
                        re.sub("(.)[^_]*_?", r"\1", key), value) \
                            for key, value in sorted(vars(args).items()) \
                                if not '/' in str(value) \
                                and not 'threads' in key
                                and not 'logdir' in key
                    )
                )
            )

    print("The logdir is: {}".format(logdir))

    # Load the data
    train_set = data.Dataset(args.train_set, args.vocab, shuffle_batches=True)
    valid_set = data.Dataset(args.valid_set, args.vocab, shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, train_set.num_tokens)

    # Train, batches
    print("Training started.")
    for i in range(args.epochs):
        while not train_set.epoch_finished():
            batch = train_set.next_batch(args.batch_size)
            network.train_batch(batch)

            # Saving embeddings
            #embeddings = network.embeddings()
            #time = datetime.datetime.now().strftime("%H%M%S")
            #file_name = logdir + '/embeddings_' + time + '.csv'
            #embeddings_to_write = '\n'.join(
            #    [','.join([str(round(j, 6)) for j in i]) for i in embeddings])
            #with open(file_name, 'w') as f:
            #    f.write(embeddings_to_write + '\n')
        accuracy = network.evaluate('valid', valid_set, args.batch_size)
        print("Accuracy on valid set after epoch {}: {:.2f}".format(
                                            i + 1, 100 * accuracy))
    print("Training finished.")

    # Save model
    model_path = network.save(logdir + '/model')
    print('Saved model path: ', model_path)

    if args.test_set:
        network = NetworkPredict()
        network.load(model_path)
        test = data.Dataset(args.test_set, args.vocab, test=True)
        p = network.predict('test', test)
        for i in p[:10]:
            print(i)

