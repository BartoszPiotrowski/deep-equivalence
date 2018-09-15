import tensorflow as ts
import sys
sys.path.append('')
from utils.parse import parse


formula = 's(o(o(X,Y),r(Z,Y,Z)),t(o(Z,Y),Z))'
parsed = parse(formula)
print(parsed)

def initialize_weights(predicates, units):
    for p in predicates:
        arity = predicates[p]




# define dense layers for each symbol
# define combining network
# define loss
# define training

# initialize weights

# for all examples
#   construct a computational tree
#   do forward and backward pass

# test
