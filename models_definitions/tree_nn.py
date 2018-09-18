import random
import torch
import sys
sys.path.append('.')
from utils import parse

DIM_IN_OUT, DIM_H = 20, 20, 20

# TODO this should be read from file
PREDICATES_WITH_ARITIES = {
    'e' = 0,
    'o' = 2,
    's' = 3,
    'b' = 2,
    'k' = 2,
    'a' = 3,
    't' = 2,
    'l' = 3,
    'r' = 3
}

# TODO this should be read from file
VARIABLES = [X, Y, Z, U, W]

class ComponentNetwork(torch.nn.Module):
    def __init__(self, predicate_name, predicate_arity, (dim_in_out, dim_h)):
        super(ComponentNetwork, self).__init__()

        self.predicate_name = predicate_name
        self.predicate_arity = predicate_arity
        self.model = torch.nn.Sequential(
            torch.nn.Linear(predicate_arity * dim_in_out, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_in_out),
        )

    def forward(self, x):
        return self.model(x)


class FormulaNN(torch.nn.Module):
    def __init__(self, predicates_with_arities, variables):
        """
        Define and instantiate small component networks for each predicate.
        """
        super(FormulaNN, self).__init__()

        self.components = {} # Dictionary for instances of ComponentNetwork
        for pred in predicates_with_arities:
            self.components[pred] = ComponentNetwork(
                pred,
                predicates_with_arities[pred],
                (DIM_IN_OUT, DIM_H)
            )

    def tree(self, formula):
        if formula[1]:
            x = torch.cat([tree(c) for c in formula[1]])
            return self.components[formula[0]](x)
        else
            return self.components['variable'](formula[0])
            # TODO case of constant e

    def forward(self, formula):
        """
        For given formula compute forward pass in tree-like network which shape
        corresponds to the shape of the formula.
        Formula is assumed to be of the form [predicate, list_of_children]
        """
        y_pred = tree(formula)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

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
