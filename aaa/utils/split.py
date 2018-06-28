import argparse, random

parser = argparse.ArgumentParser(
    description="Split file with training examples into three balanced.")
parser.add_argument('filename', type=str, help="Path to file.")
parser.add_argument('--train', type=float, help="Train set size.")
parser.add_argument('--valid', type=float, help="Validation set size.")
parser.add_argument('--test', type=float, help="Test set size.")
args = parser.parse_args()
assert args.train + args.valid + args.test == 1.
examples = open(args.filename).read().split('\n')
examples = [e for e in examples if e] # remove empty lines
examples = list(set(examples)) # remove duplicates
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
    f.write('\n'.join(examplesTrain))
with open(args.filename + '.valid', 'w') as f:
    f.write('\n'.join(examplesValid))
with open(args.filename + '.test', 'w') as f:
    f.write('\n'.join(examplesTest))


