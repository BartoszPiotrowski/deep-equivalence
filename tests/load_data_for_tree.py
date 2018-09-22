import sys
sys.path.append('')
from utils.dataset import DatasetTreeNN


d = DatasetTreeNN('data/split/equiv.train')
print(len(d))
p = d._permutation[:10]
for i in p:
    print(d.terms_L[i], d.terms_R[i], d.labels[i])

print(d.__next__())
print(d._current_index)
print(d.__next__())
print(d._current_index)
print(d.__next__())
print(d._current_index)
print(d.__next__())
print(d._current_index)

for i in d:
    print(i)
