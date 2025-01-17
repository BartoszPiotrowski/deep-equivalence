# deep-equivalence

## Requirements

`Python 3.6.6` with `Tensorflow 1.10` package installed. If you have other
versions, please, follow the instructions in
[installation.md](installation.md) -- otherwise something may not work properly.


**The repo contains 3 pretrained models (in
[models_pretrained/](models_pretrained/)) you can query -- see the last section.
If you want to reproduce the preparation of the training data and training of
the models -- follow the next two sections.**


## Praparation of data for experiments

File `data/from_Chad/quadsforaim.p` contains data as received from Chad. In order
to preprocess them to the form appropriate for the experiments do the following:
1. Run `python3 utils/prepare_data.py data/from_Chad/quadsforaim.p > data/equiv`
2. Run `python3 utils/split.py data/equiv --train 0.5 --valid 0.3 --test 0.2`
3. Move split data to its place: `mkdir data/split; mv data/equiv.* data/split`

Optionally you can:

4. Augment training examples by doing permutations:
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

The implemented machine learning model is a straightforward bidirectional RNN
with GRU cells. To train it run:
```
mkdir models_pretrained
```
```
python3 models_definitions/bidir_rnn.py \
	--train_set data/split/equiv.train \
	--valid_set data/split/equiv.valid \
	--model_path models_pretrained/basic \
	--vocab data/vocab \
	--batch_size 128 \
	--epochs 64 \
	--embed_dim 8 \
	--rnn_cell_dim 16 \
	--num_dense_layers 2 \
	--dense_layer_units 32
```
The accuracy of this model on the validation set should go up to `88%--91%`.

Now let's see what happens when variables in training and validation set are
randomly renamed.
```
python3 models_definitions/bidir_rnn.py \
	--train_set data/split/equiv_renamed.train \
	--valid_set data/split/equiv_renamed.valid \
	--model_path models_pretrained/renamed_train_and_valid \
	--vocab data/vocab \
	--batch_size 128 \
	--epochs 64 \
	--embed_dim 8 \
	--rnn_cell_dim 16 \
	--num_dense_layers 2 \
	--dense_layer_units 32
```
The accuracy of this model on the validation set doesn't go higher than `86%`.
This means that the model make use of the ordering of variables -- when variables
in formulae appear in a consistent order, `X, Y, Z, ...`, it makes learning for
the model easier. This can be considered as an unintended behaviour or a "leak
in the data" as we want our model to "really understand" the "notion of
equivalence." On the other hand we can argue this is alright.

We can also check what happens when variables in validation set are randomly
renamed, whereas in the training set variables appear in consistent order across
the formulae.
```
python3 models_definitions/bidir_rnn.py \
	--train_set data/split/equiv.train \
	--valid_set data/split/equiv_renamed.valid \
	--model_path models_pretrained/renamed_valid \
	--vocab data/vocab \
	--batch_size 128 \
	--epochs 64 \
	--embed_dim 8 \
	--rnn_cell_dim 16 \
	--num_dense_layers 2 \
	--dense_layer_units 32
```
Accuracy is `< 77%`. This means that model is deceived -- it learned to use the
information coming from consistent ordering of variables, but this was
unnapplicable on the validation.

In the picture below we see training stats of the above 3 models:

![Training stats 1](pictures/tensorboard1.png?raw=true)


Finally, let's train the model on training data augmented by all possible
renamings of variables. Original training set has 16937 examples whereas after
augmenting we see as many as 2853850 examples -- so the training time will be
roughly 170 times longer.

```
python3 models_definitions/bidir_rnn.py \
	--train_set data/split/equiv_augmented.train \
	--valid_set data/split/equiv_renamed.valid \
	--model_path models_pretrained/augmented_train \
	--vocab data/vocab \
	--batch_size 128 \
	--epochs 64 \
	--embed_dim 8 \
	--rnn_cell_dim 16 \
	--num_dense_layers 2 \
	--dense_layer_units 32
```

In the picture below we see training stats of this model:

![Training stats 2](pictures/tensorboard2.png?raw=true)


## Quering the trained models

Having trained some models we can query them with some new pairs of formulae.
Let's choose one of the models from `models_pretrained/models` and do prediction:
```
python3 utils/predict.py \
	--model models_pretrained/models/basic \
	--pairs models_pretrained/data/split/equiv_no_labels.test
```

Each number is a probability of the pair being equivalent, according to the model.
(By adding a flag `--discrete` the model will output 0s and 1s instead of
probabilities.)
