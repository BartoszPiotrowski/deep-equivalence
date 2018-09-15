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
python3 utils/rename.py data/split/equiv.train --variables 'XYZUW' > \
						data/split/equiv_renamed.train
python3 utils/rename.py data/split/equiv.valid --variables 'XYZUW' > \
						data/split/equiv_renamed.valid
python3 utils/rename.py data/split/equiv.test --variables 'XYZUW' > \
						data/split/equiv_renamed.test
```

## Training a model

The implemented machine learning model is a straightforward bidirectional RNN.
To train it run:
```
mkdir trained_models
python3 models/bidir_rnn.py \
	--train_set data/split/equiv.train \
	--valid_set data/split/equiv.valid \
	--model_path trained_models/basic_split \
	--vocab data/vocab \
	--batch_size 128 \
	--epochs 32 \
	--embed_dim 8 \
	--rnn_cell_dim 16 \
	--num_dense_layers 2 \
	--dense_layer_units 32
```
The accuracy of this model on the validation set should go up to `90%--91%`.

Now let's see what happens when variables in training and validation set are
randomly renamed.
```
python3 models/bidir_rnn.py \
	--train_set data/split/equiv_renamed.train \
	--valid_set data/split/equiv_renamed.valid \
	--model_path trained_models/basic_split \
	--vocab data/vocab \
	--batch_size 128 \
	--epochs 32 \
	--embed_dim 8 \
	--rnn_cell_dim 16 \
	--num_dense_layers 2 \
	--dense_layer_units 32
```
The accuracy of this model on the validation set should go not higher than `85%`.
This means that the model make use of the ordering of variables -- when variables
in formulae appear in a consistent order, `X, Y, Z, ...`, it makes learning for
the model easier. This can be considered as a unintended behaviour or a "leak
in the data" as we want our model to "really understand" the "notion of
equivalence." On the other hand we can argue this is alright.

We can also check what happens when variables in validation set are randomly
renamed, whereas in the training set variables appear in consistent order across
the formulae.
```
python3 models/bidir_rnn.py \
	--train_set data/split/equiv.train \
	--valid_set data/split/equiv_renamed.valid \
	--model_path trained_models/basic_split \
	--vocab data/vocab \
	--batch_size 128 \
	--epochs 32 \
	--embed_dim 8 \
	--rnn_cell_dim 16 \
	--num_dense_layers 2 \
	--dense_layer_units 32
```
Accuracy is `< 77%`. This means that model is deceived -- it learned to use the
information coming from consistent ordering of variables, but this was
unnapplicable on the validation.

Finally, let's train the model on training data augmented by all possible
renamings of variables. Original training set has 16937 examples whereas after
augmenting we see as many as 2853850 examples. Therefore let's increase number
of training epoch from 32 to 128.
```
python3 models/bidir_rnn.py \
	--train_set data/split/equiv_augmented.train \
	--valid_set data/split/equiv_renamed.valid \
	--model_path trained_models/basic_split \
	--vocab data/vocab \
	--batch_size 128 \
	--epochs 32 \
	--embed_dim 8 \
	--rnn_cell_dim 16 \
	--num_dense_layers 2 \
	--dense_layer_units 32
```


## Quering a trained model

```
python3 models/predict.py --model ... --pairs data/split/equiv_no_labels.test
```
