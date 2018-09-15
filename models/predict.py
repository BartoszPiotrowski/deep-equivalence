#!/usr/bin/env python3
import tensorflow as tf
import sys
sys.path.append('')
from utils.dataset import Dataset

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
            self.logits = tf.get_collection(
                'end_points/logits')[0]

            # Load the graph weights
            self.saver.restore(self.session, path)

    def predict(self, dataset, discrete=True):
        tokens_ids_1, formulae_lens_1, \
        tokens_ids_2, formulae_lens_2, \
            = dataset.test()
        if discrete:
            return self.session.run(self.predictions,
                             {self.formulae_lens_1: formulae_lens_1,
                              self.tokens_ids_1: tokens_ids_1,
                              self.formulae_lens_2: formulae_lens_2,
                              self.tokens_ids_2: tokens_ids_2})
        else:
            return self.session.run(self.logits,
                             {self.formulae_lens_1: formulae_lens_1,
                              self.tokens_ids_1: tokens_ids_1,
                              self.formulae_lens_2: formulae_lens_2,
                              self.tokens_ids_2: tokens_ids_2})[:,1]


if __name__ == "__main__":
    import argparse
    import os


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
    parser.add_argument(
        '--discrete',
        action='store_true',
        help="By default the model returns probabilities; setting this flag \
            causes returning 0s and 1s.")
    args = parser.parse_args()

    all_files = os.listdir(args.model)
    [meta] = [f for f in all_files if '.meta' in f]
    prefix = meta.split('.')[0]
    model_with_prefix = args.model + '/' + prefix
    network = NetworkPredict()
    network.load(model_with_prefix)
    test = Dataset(args.pairs, args.vocab, test=True)
    p = network.predict(test, args.discrete)
    for i in p:
        print(i)
