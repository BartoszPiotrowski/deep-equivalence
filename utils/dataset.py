#!/usr/bin/env python3
import numpy as np

class Dataset:
    def __init__(
            self,
            formulae_filename, # TODO remove commas?
            vocab_filename,
            shuffle_batches=True):
        self.formulae_1 = []
        self.formulae_2 = []
        self.labels = []
        with open(formulae_filename, 'r') as formulae:
            for line in formulae:
                l, f1, f2 = line.strip('\n').split(' ')
                self.formulae_1.append(f1)
                self.formulae_2.append(f2)
                self.labels.append(l)
        with open(vocab_filename, 'r') as vocab:
            self.vocab = vocab.read().splitlines()
        self.vocab_map = {self.vocab[i]: i + 1 for i in range(len(self.vocab))}
        self.seqs_1 = [[self.vocab_map[t] for t in f] for f in self.formulae_1]
        self.seqs_2 = [[self.vocab_map[t] for t in f] for f in self.formulae_2]
        self.formulae_lens_1 = [len(f) for f in self.formulae_1]
        self.formulae_lens_2 = [len(f) for f in self.formulae_2]
        self.shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self)) \
                if self.shuffle_batches else np.arange(len(self))

    def __len__(self):
        return len(self.formulae_1)

    def pad(self, sequences, length, pad=0):
        padded_sequences = []
        for s in sequences:
            assert len(s) <= length
            padded_sequences.append(s + [pad] * (length - len(s)))
        return padded_sequences

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = \
            self._permutation[:batch_size], self._permutation[batch_size:]
        l1 = [self.formulae_lens_1[i] for i in batch_perm]
        l2 = [self.formulae_lens_2[i] for i in batch_perm]
        max_len = max(l1 + l2)
        seqs_1 = np.array(self.pad([self.seqs_1[i] for i in batch_perm], max_len))
        seqs_2 = np.array(self.pad([self.seqs_2[i] for i in batch_perm], max_len))
        labels = [self.labels[i] for i in batch_perm]
        return seqs_1, seqs_2, labels

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self.sequences)) \
                    if self.shuffle_batches else np.arange(len(self.sequences))
            return True
        return False
