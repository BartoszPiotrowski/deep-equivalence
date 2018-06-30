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
                f1, f2, l = line.strip('\n').split(' ')
                self.formulae_1.append(f1)
                self.formulae_2.append(f2)
                self.labels.append(l)
        with open(vocab_filename, 'r') as vocab:
            self.vocab = vocab.read().splitlines()
        self.vocab_map = {self.vocab[i]: i + 1 for i in range(len(self.vocab))}
        self.seqs_1 = [[self.vocab_map[t] for t in f] for f in self.formulae_1]
        self.seqs_2 = [[self.vocab_map[t] for t in f] for f in self.formulae_2]
        self.formulae_lengths_1 = [len(f) for f in self.formulae_1]
        self.formulae_lengths_2 = [len(f) for f in self.formulae_2]
        self.shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self.sequences)) \
                if self.shuffle_batches else np.arange(len(self.sequences))

    def all_data(self):
        return self.sequences, self.labels

    def padding(sequences, length, pad=0):
        padded_sequences = []
        for s in sequences:
            assert len(s) <= length
            padded_sequences.append(s + [pad] * (length - len(s)))
        return padded_sequences

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = \
            self._permutation[:batch_size], self._permutation[batch_size:]
        max_length = max(self.formulae_lengths_1[batch_perm] +
                         self.formulae_lengths_2[batch_perm])
        sequences = padding(self.sequences[batch_perm], max_length)
        labels = self.labels[batch_perm]
        return sequences, labels

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self.sequences)) \
                    if self.shuffle_batches else np.arange(len(self.sequences))
            return True
        return False
