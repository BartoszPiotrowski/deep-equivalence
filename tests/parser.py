import sys
sys.path.append('..')
from utils.parse import parse

# Test:
print(parse('k(X,t(t(b(o(X,Y),Z),o(Y,X)),U))'))
print(parse('b(t(X,t(Y,Z)),Y)'))
print(parse('b(Y,o(X,Z))'))
print(parse('b(Y,o(X,Y,Z),o(X,Z,W,U))'))
