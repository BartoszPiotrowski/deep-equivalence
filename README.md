# deep-equivalence

## Requirements:
`python3` with `tensorflow` package installed.

## Praparation of data for experiments:

File `data/from-Chad/quadsforaim.p` contains data as received from Chad. In order
to preprocess them to the form appropriate for experiments do the following:
1. Run `python3 utils/prepare-data.py data/from-Chad/quadsforaim.p > data/equiv`
2. Run `python3 utils/split.py data/equiv --train 0.5 --valid 0.3 --test 0.2`
3. Move split data to its place: `mkdir data/split; mv data/equiv.* data/split`

Optionally you can:

4. Augment training examples by doing permutation:
```
python3 utils/augment.py data/split/equiv.train --variables 'XYZUW' --reverse \
		> data/split/equiv_augmented.train
```
5. Randomly rename variables in `data/split/equiv.valid` and
   `data/split/equiv.test`:
```
python3 utils/rename.py data/split/equiv.valid --variables 'XYZUW' > \
						data/split/equiv_renamed.valid
mv data/split/equiv_renamed.valid data/split/equiv.valid
python3 utils/rename.py data/split/equiv.test --variables 'XYZUW' > \
						data/split/equiv_renamed.test
mv data/split/equiv_renamed.test data/split/equiv.test
```

## Training a model

To train the model for recognizing equivalence (simple bidirectional RNN), run:
```
python3 models/bidir-rnn.py \
	--train_set data/split/equiv.train \
	--valid_set data/split/equiv.valid \
	--test_set data/split/equiv.test \
	--vocab data/vocab \
	--batch_size 64 \
	--epochs 32 \
	--embed_dim 4 \
	--rnn_cell_dim 8 \
	--num_dense_layers 2 \
	--dense_layer_units 16
```
