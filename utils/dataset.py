#!/usr/bin/env python3
import numpy as np


class Dataset: # TODO rename it to DatasetRNN and make superclass Dataset
    def __init__(self,
                terms_filename,  # TODO remove commas?
                vocab_filename,
                shuffle=True,
                test=False):
        self.terms_L = []
        self.terms_R = []
        self.labels = []
        with open(terms_filename, 'r') as terms:
            if test:
                for line in terms:
                    t1, t2 = line.strip('\n').split(' ')
                    self.terms_L.append(t1)
                    self.terms_R.append(t2)
            else:
                for line in terms:
                    l, t1, t2 = line.strip('\n').split(' ')
                    self.terms_L.append(t1)
                    self.terms_R.append(t2)
                    self.labels.append(l)
        with open(vocab_filename, 'r') as vocab:
            self.vocab = vocab.read().splitlines()
        self.num_tokens = len(self.vocab) + 1  # + 1 because of padding with 0s
        self.vocab_map = {self.vocab[i]: i + 1 for i in range(len(self.vocab))}
        self.seqs_1 = [[self.vocab_map[t] for t in f] for f in self.terms_L]
        self.seqs_2 = [[self.vocab_map[t] for t in f] for f in self.terms_R]
        self.terms_lens_1 = [len(f) for f in self.terms_L]
        self.terms_lens_2 = [len(f) for f in self.terms_R]
        self.shuffle = shuffle
        self._permutation = np.random.permutation(len(self)) \
            if self.shuffle else np.arange(len(self))

    def __len__(self):
        return len(self.terms_L)

    def pad(self, sequences, length, pad_symbol=0):
        padded_sequences = []
        for s in sequences:
            assert len(s) <= length
            padded_sequences.append(s + [pad_symbol] * (length - len(s)))
        return padded_sequences

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = \
            self._permutation[:batch_size], self._permutation[batch_size:]
        lens_1 = [self.terms_lens_1[i] for i in batch_perm]
        lens_2 = [self.terms_lens_2[i] for i in batch_perm]
        max_len = max(lens_1 + lens_2)
        seqs_1 = np.array(self.pad([self.seqs_1[i]
                                    for i in batch_perm], max_len))
        seqs_2 = np.array(self.pad([self.seqs_2[i]
                                    for i in batch_perm], max_len))
        labels = [self.labels[i] for i in batch_perm]
        return seqs_1, lens_1, seqs_2, lens_2, labels

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self)) \
                if self.shuffle else np.arange(len(self))
            return True
        return False

    def test(self):
        max_len = max(self.terms_lens_1 + self.terms_lens_2)
        seqs_1 = np.array(self.pad(self.seqs_1, max_len))
        seqs_2 = np.array(self.pad(self.seqs_2, max_len))
        return seqs_1, self.terms_lens_1, seqs_2, self.terms_lens_2


class DatasetTreeNN:
    def __init__(self,
                terms_filename,  # TODO remove commas?
                functs_with_arits_filename,
                consts_vars_filename,
                shuffle=True,
                predict_mode=False):

        self.terms_L = []
        self.terms_R = []
        self.labels = []

        with open(terms_filename, 'r') as terms:
            if predict_mode:
                for line in terms:
                    t1, t2 = line.strip('\n').split(' ')
                    self.terms_L.append(t1)
                    self.terms_R.append(t2)
            else:
                for line in terms:
                    l, t1, t2 = line.strip('\n').split(' ')
                    self.terms_L.append(t1)
                    self.terms_R.append(t2)
                    self.labels.append(l)

        with open(functs_with_arits_filename, 'r') as f:
            lines = f.read().splitlines()
            self.functs_with_arits = \
                dict((f_a.split(' ')[0], int(f_a.split(' ')[1])) for f_a in lines)
        with open(consts_vars_filename, 'r') as f:
            self.vars_consts = f.read().splitlines()

        self.shuffle = shuffle
        self._permutation = np.random.permutation(len(self)) \
            if self.shuffle else np.arange(len(self))
        self._current_index = 0 # TODO do it in Pythonic way


    def __len__(self):
        return len(self.terms_L)


    def __iter__(self):
        return self


    def __next__(self): # TODO do it in Pythonic way
        if self._current_index >= len(self):
            if self.shuffle:
                self._permutation = np.random.permutation(len(self))
            raise StopIteration
        else:
            i = self._permutation[self._current_index]
            self._current_index += 1
            return self.labels[i], self.terms_L[i], self.terms_R[i]

