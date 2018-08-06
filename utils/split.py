#!/usr/bin/env python3
# usage: python3 utils/split.py data/equiv --train 0.5 --valid 0.2 --test 0.3
import argparse, random

parser = argparse.ArgumentParser(
    description="Split file with training examples into three balanced.")
parser.add_argument('filename', type=str, help="Path to file.")
parser.add_argument('--train', type=float, help="Train set size.")
parser.add_argument('--valid', type=float, help="Validation set size.")
parser.add_argument('--test', type=float, help="Test set size.")
args = parser.parse_args()
assert args.train + args.valid + args.test == 1.
with open(args.filename, 'r') as f:
    examples = f.read().splitlines()
examples = [e for e in examples if e] # remove empty lines
examples = [' '.join([i[0], i[1], i[2]]) if i[1] < i[2] else \
            ' '.join([i[0], i[2], i[1]]) \
                for i in [e.split(' ') for e in examples]] # order pairs
examples = list(set(examples)) # remove duplicates
with open(args.filename + '_no_duplicates', 'w') as f:
    f.write('\n'.join(examples) + '\n')
random.shuffle(examples)
examplesPos = [e for e in examples if e.split(' ')[0] == '1']
examplesNeg = [e for e in examples if e.split(' ')[0] == '0']
nPos = len(examplesPos)
nNeg = len(examplesNeg)
assert nPos + nNeg == len(examples)
nTrainPos = round(args.train * nPos)
nTrainNeg = round(args.train * nNeg)
nTestPos = round(args.test * nPos)
nTestNeg = round(args.test * nNeg)
examplesPosTrain = examplesPos[:nTrainPos]
examplesNegTrain = examplesNeg[:nTrainNeg]
examplesPosValid = examplesPos[nTrainPos: - nTestPos]
examplesNegValid = examplesNeg[nTrainNeg: - nTestNeg]
examplesPosTest = examplesPos[- nTestPos:]
examplesNegTest = examplesNeg[- nTestNeg:]
examplesTrain = examplesPosTrain + examplesNegTrain
examplesValid = examplesPosValid + examplesNegValid
examplesTest = examplesPosTest + examplesNegTest
random.shuffle(examplesTrain)
random.shuffle(examplesValid)
random.shuffle(examplesTest)
assert not set(examplesTrain) & set(examplesValid)
assert not set(examplesValid) & set(examplesTest)
assert not set(examplesTest) & set(examplesTrain)
with open(args.filename + '.train', 'w') as f:
    f.write('\n'.join(examplesTrain) + '\n')
with open(args.filename + '.valid', 'w') as f:
    f.write('\n'.join(examplesValid) + '\n')
with open(args.filename + '.test', 'w') as f:
    f.write('\n'.join(examplesTest) + '\n')

