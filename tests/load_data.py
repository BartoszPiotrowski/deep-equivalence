import sys
sys.path.append('')
from utils import dataset as ds


d = ds.Dataset('data/split/equiv.train', 'data/vocab.txt')
print(len(d))
p = d._permutation[:10]
for i in p:
    print(d.formulae_1[i], d.formulae_2[i], d.labels[i])
n = d.next_batch(10)
print(n)
print(d.num_tokens)

