import sys
sys.path.append('')
from utils import dataset as ds


d = ds.Dataset('data/split/equiv.train', 'data/vocab.txt')
print(len(d))
