#!/usr/bin/env python3
import tensorflow as tf
import sys
sys.path.append('')
#from models.bidir_rnn import NetworkPredict

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


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Path to a trained model file.")
    parser.add_argument(
        "--pairs",
        type=str,
        help="File with pairs of formulae for which we want to predict its \
        equivalence.")
    parser.add_argument(
        "--vocab",
        default='data/vocab',
        type=str,
        help="Path to a vocabulary file.")
    args = parser.parse_args()

    network = NetworkPredict()
    print(args.model)
    print(args.pairs)
    print(args.vocab)
    network.load(args.model)
    test = data.Dataset(args.pairs, args.vocab, test=True)
    p = network.predict('test', test)
    for i in p:
        print(i)
