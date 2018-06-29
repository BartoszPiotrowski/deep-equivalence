
class Dataset:
    def __init__(self, sequences, sequence_length, sequence_dim, shuffle_batches=True):
        self._sequences = np.zeros([sequences, sequence_length, sequence_dim], np.int32)
        self._labels = np.zeros([sequences, sequence_length], np.bool)

        for i in range(sequences):
            self._sequences[i, :, 0] = np.random.random_integers(0, max(1, sequence_dim - 1), size=[sequence_length])
            self._labels[i] = np.bitwise_and(np.cumsum(self._sequences[i, :, 0]), 1)
            if sequence_dim > 1:
                self._sequences[i] = np.eye(sequence_dim)[self._sequences[i, :, 0]]

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._sequences)) if self._shuffle_batches else np.arange(len(self._sequences))

    @property
    def sequences(self):
        return self._sequences

    @property
    def labels(self):
        return self._labels

    def all_data(self):
        return self._sequences, self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._sequences[batch_perm], self._labels[batch_perm]

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._sequences)) if self._shuffle_batches else np.arange(len(self._sequences))
            return True
        return False
