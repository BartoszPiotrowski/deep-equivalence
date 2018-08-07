## Praparation of data for experiments:
1. Run `python3 utils/prepare-data.py data/from-Chad/quadsforaim.p > data/equiv`
2. Run `python3 utils/split.py data/equiv --train 0.5 --valid 0.3 --test 0.2`
3. Move split data: `mkdir data/split; mv data/equiv.* data/split`
4. Augment training examples:
`
python3 utils/augment.py data/split/equiv.train --variables 'XYZUW' --reverse > \
		data/split/equiv_augmented.train
`
5. Randomly rename variables in `data/split/equiv.valid` and
   `data/split/equiv.test`:
`
python3 utils/rename.py data/split/equiv.valid --variables 'XYZUW' > \
							data/split/equiv.valid
python3 utils/rename.py data/split/equiv.test --variables 'XYZUW' > \
							data/split/equiv.test
`
