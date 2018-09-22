# Ideas
We want the network to be symmetric:
```
network(term1, term2) = network(term2, term1)
network(term1, term1) = 1
```
We can achieve this by not having the 'top module', just measuring the distance
of embeddings.
