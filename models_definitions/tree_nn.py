import random
import torch
import sys
sys.path.append('.')
from utils.parse import parse


class PredicteNetwork(torch.nn.Module):
    def __init__(self, predicate_name, predicate_arity,
                 dim_in_out=64, dim_h=64):
        super(PredicteNetwork, self).__init__()
        self.predicate_name = predicate_name
        self.predicate_arity = predicate_arity
        self.model = torch.nn.Sequential(
            torch.nn.Linear(predicate_arity * dim_in_out, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_in_out),
        )

    def forward(self, x):
        return self.model(x)


class ConstantNetwork(torch.nn.Module):
    # TODO bias=True?
    def __init__(self, predicate_name, predicate_arity, dim_out=64):
        super(ConstantNetwork, self).__init__()
        self.predicate_name = predicate_name
        self.predicate_arity = predicate_arity
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, dim_out),
        )

    def forward(self): # TODO can we ommit the second argument?
        return self.model()


class VariableNetwork(torch.nn.Module):
    def __init__(self, dim_in=5, dim_h=64, dim_out=64):
        super(VariableNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_out),
        )

    def forward(self, x):
        return self.model(x)


class EqualityNetwork(torch.nn.Module):
    def __init__(self, dim_in=64, dim_h=64):
        super(EqualityNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2 * dim_in, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, 1),
        )

    def forward(self, x):
        return self.model(x)


def loss(y_pred, y):
    return (y_pred - y).pow(2).item() # TODO not pow(2) but |.|


def one_hot(elem, elems):
    if isinstance(elems, int):
        assert 0 <= elem < elems
        elems = range(elems)
    else:
        assert len(set(elems)) == len(elems)
    return [1 if e ==  elem else 0 for e in elems]

def variables_as_tensors(all_variables):
    return {v: torch.tensor([one_hot(v, all_variables)], dtype=torch.float)
                        for v in all_variables}

############ TEST ###############################################
term = 'b(Y,o(X,Z))'
term_2 = 'k(X,t(t(b(o(X,Y),Z),o(Y,X)),U))'
b_net = PredicteNetwork('b', 2)
o_net = PredicteNetwork('b', 2)
var_net = VariableNetwork()
# TODO this should be read from file
VARIABLES = ['X', 'Y', 'Z', 'U', 'W']
PREDS_WITH_ARITIES = {
    'e': 0,
    'o': 2,
    's': 3,
    'b': 2,
    'k': 2,
    'a': 3,
    't': 2,
    'l': 3,
    'r': 3
}

vs = variables_as_tensors(VARIABLES)
#print(loss(torch.tensor([0.0], dtype=torch.float),
#           torch.tensor([1.0], dtype=torch.float)))

#print(one_hot(4, 10))
#print(one_hot('a', ['a', 'b', 'c']))

#tree_net = b_net(
#    torch.cat(
#        (var_net(vs['Y']), o_net(torch.cat((var_net(vs['X']),var_net(vs['Z'])),
#                                           -1))),
#        -1))
#print(tree_net)

parsed_term = parse(term)
parsed_term_2 = parse(term_2)
print(parsed_term_2)

preds_with_arities = PREDS_WITH_ARITIES
components = {} # Dictionary for instances of PredicteNetwork and ConstantNetwork
for pred in preds_with_arities:
    components[pred] = \
        PredicteNetwork(pred, preds_with_arities[pred]) \
                if preds_with_arities[pred] else \
        ConstantNetwork(pred, preds_with_arities[pred])
components['VAR'] = VariableNetwork()

def tree(term):
    if len(term) > 1:
        x = torch.cat([tree(arg) for arg in term[1]], -1)
        return components[term[0]](x)
    else:
        if term[0] in VARIABLES:
            return components['VAR'](vs[term[0]])
        else: # case of constant
            assert not preds_with_arities[term[0]]
            return components[term[0]](torch.tensor())

#print(tree(parsed_term))
#print(tree(parsed_term_2))
eq = EqualityNetwork()
#print(eq(torch.cat((tree(parsed_term), tree(parsed_term_2)), -1)))
l = loss(1, eq(torch.cat((tree(parsed_term), tree(parsed_term_2)), -1)))
print(l)


sys.exit()
############ TEST ENDED #########################################


class TermNN(torch.nn.Module):
    def __init__(self, preds_with_arities, variables):
        """
        Define and instantiate small component networks for each predicate.
        """
        super(TermNN, self).__init__()

        self.components = {} # Dictionary for instances of PredicteNetwork
        for pred in preds_with_arities:
            self.components[pred] = PredicteNetwork(
                pred,
                preds_with_arities[pred]
            )

    def tree(self, term):
        if term[1]:
            x = torch.cat([tree(c) for c in term[1]], -1)
            return self.components[term[0]](x)
        else:
            return self.components['variable'](term[0])
            # TODO case of constant e

    def forward(self, term):
        """
        For given term compute forward pass in tree-like network which shape
        corresponds to the shape of the term. Term is assumed to be of the parsed
        form [predicate, list_of_arguments].
        """
        y_pred = tree(term)
        return y_pred



#####################################

DIM_IN_OUT, DIM_H = 64, 64


# TODO this should be read from file
VARIABLES = ['X', 'Y', 'Z', 'U', 'W']

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# TODO remove it
test_example = 'k(X,t(t(b(o(X,Y),Z),o(Y,X)),U))'

# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
