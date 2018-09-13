import sys
sys.path.append('')
from models.bidir-rnn import NetworkPredict
from utils import dataset as data

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re


    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Path to a trained model file.")
    parser.add_argument(
        "--pairs",
        type=str,
        help="File with pairs of formulae for which we want to predict its
        equivalence.")
    parser.add_argument(
        "--vocab",
        default='data/vocab',
        type=str,
        help="Path to a vocabulary file.")
    args = parser.parse_args()

    network = NetworkPredict()
    network.load(args.model)
    test = data.Dataset(args.pairs, args.vocab, test=True)
    p = network.predict('test', test)
    for i in p:
        print(i)
