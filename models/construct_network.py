#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


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

            # RNN Cell
            if args.rnn_cell == "LSTM":
                rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
            elif args.rnn_cell == "GRU":
                rnn_cell = tf.nn.rnn_cell.GRUCell
            else:
                raise ValueError("Unknown rnn_cell {}".format(args.rnn_cell))

            # Token embeddings
            token_embeddings = tf.get_variable(
                "token_embeddings", shape=[num_tokens, args.embed_dim],
                dtype=tf.float32)
            inputs_1 = tf.nn.embedding_lookup(token_embeddings, self.tokens_ids_1)
            inputs_2 = tf.nn.embedding_lookup(token_embeddings, self.tokens_ids_2)

            # Computation
            _, (state_fwd_1, state_bwd_1) = \
                    tf.nn.bidirectional_dynamic_rnn(
                        rnn_cell(args.rnn_cell_dim),
                        rnn_cell(args.rnn_cell_dim),
                        inputs_1,
                        sequence_length=self.formulae_lens_1,
                        dtype=tf.float32)
            _, (state_fwd_2, state_bwd_2) = \
                    tf.nn.bidirectional_dynamic_rnn(
                        rnn_cell(args.rnn_cell_dim),
                        rnn_cell(args.rnn_cell_dim),
                        inputs_2,
                        sequence_length=self.formulae_lens_2,
                        dtype=tf.float32)
            state_1 = tf.concat([state_fwd_1, state_bwd_1]) # TODO axis OK?
            state_2 = tf.concat([state_fwd_2, state_bwd_2]) # TODO axis OK?
            state = tf.concat([state_1, state_2], axis=2) # TODO axis OK?
            # TODO parameter in place of 64
            layer_1 = tf.layers.dense(state, 64, activation=tf.nn.relu)
            logits = tf.layers.dense(layer_1, self.LABELS, name='logits')
            self.predictions = tf.argmax(logits, axis=1, name='predictions')

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(
                self.labels, logits)
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(
                loss, global_step=global_step, name="training")

            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(
                self.labels, self.predictions)
            self.current_loss, self.update_loss = tf.metrics.mean(loss))
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
            formulae_lens, tokens_ids, labels = train.next_batch(batch_size)
            self.session.run(self.reset_metrics)
            self.session.run([self.training,
                              self.summaries["train"]],
                             {self.formulae_lens: formulae_lens,
                              self.tokens_ids: tokens_ids,
                              self.labels: labels})

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            formulae_lens, tokens_ids = dataset.next_batch(batch_size)
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.formulae_lens: formulae_lens,
                              self.tokens_ids: tokens_ids,
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
        "--rnn_cell_dim",
        default=64,
        type=int,
        help="RNN cell dimension.")
    parser.add_argument(
        "--threads",
        default=1,
        type=int,
        help="Maximum number of threads to use.")
    parser.add_argument("--embed_dim", default=128, type=int,
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
    train = tptp_dataset.Dataset("data/split.train", max_sentences=5000)
    dev = tptp_dataset.Dataset("data/split.valid", shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, train.num_tokens)

    # Train
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size)
        accuracy = network.evaluate("dev", dev, args.batch_size)
        print("{:.2f}".format(100 * accuracy))
