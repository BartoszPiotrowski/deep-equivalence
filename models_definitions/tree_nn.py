import torch
import random
import argparse
import sys
sys.path.append('.')
from utils.parse import parse
from utils.tools import one_hot
from utils.dataset import DatasetTreeNN


class FunctionNetwork(torch.nn.Module):
    def __init__(self, function_name, function_arity,
                 dim_in_out=32, dim_h=32):
        super(FunctionNetwork, self).__init__()
        self.function_name = function_name
        self.function_arity = function_arity
        self.model = torch.nn.Sequential(
            torch.nn.Linear(function_arity * dim_in_out, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_in_out),
        )

    def forward(self, x):
        return self.model(x)


class VarConstNetwork(torch.nn.Module):
    def __init__(self, dim_in=8, dim_h=32, dim_out=32):
        super(VarConstNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_out),
        )

    def forward(self, x):
        return self.model(x)


class EqualityNetwork(torch.nn.Module):
    def __init__(self, dim_in=32, dim_h=32):
        super(EqualityNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2 * dim_in, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, 1),
            # TODO add sigmoid
        )

    def forward(self, x): # TODO this repeats; make a super-class
        return self.model(x)


class TreeNN:
    def __init__(
        self,
        functs_with_arits,
        vars_consts_as_tensors,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(params_of_modules, lr=0.001, momentum=0.8)):
        # TODO add parameter for controlling layer size

        self.functs_with_arits = functs_with_arits
        self.vars_consts_as_tensors = vars_consts_as_tensors
        self.n_vars_consts = len(vars_consts_as_tensors)

        # instantiate modules for all function symbols
        self.modules = {}
        # TODO add parameter for controlling layer size
        for func in self.functs_with_arits:
            modules[func] =  FunctionNetwork(func, self.functs_with_arits[func])
        modules['VARCONST'] = VarConstNetwork(dim_in=self.n_vars_consts)
        modules['EQUALITY'] = EqualityNetwork()
        self.loss = loss
        self.optimizer = optimizer
        return modules


    def parameters(self):
        parameters = []
        for m in self.modules:
            parameters.extend(self.modules[m].parameters())
        return parameters


    def tree(self, term):
        if len(term) > 1:
            x = torch.cat([self.tree(t) for t in term[1]], -1)
            return modules[term[0]](x)
        else:
            return modules['VARCONST'](self.vars_consts_as_tensors[term[0]])


    def forward(self, example):
        term_L, term_R = example
        return self.modules['EQUALITY'](torch.cat(
            self.tree(term_L, self.modules, self.vars_consts_as_tensors),
            self.tree(term_R, self.modules, self.vars_consts_as_tensors),
            -1))


    def train_one_example(self, example_with_label):
        label, example = example_with_label
        pred = self.forward(example)
        loss = self.loss(pred, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), pred.argmax().item()


    def train(self, examples, epochs=10, verbose=True): # examples with labels
        for e in range(epochs):
            losses = []
            preds = []
            for example in examples:
                loss, pred = self.forward(example)
                losses.append(loss)
                preds.append(pred)
            N = len(examples)
            labels = [e[0].item() for e in examples]
            loss_avg = sum(losses) / N
            accuracy = sum(preds[i] == labels[i] for i in range(N)) / N
            if verbose:
                print("Loss on training {}. Accuracy on training {}.".format(
                    loss_avg, accuracy))


    def predict(inputs, model):
        return [model(i).argmax().item() for i in inputs]



############ TEST ###############################################


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_set",
    type=str,
    help="Path to a training set.")
parser.add_argument(
    "--valid_set",
    type=str,
    help="Path to a validation set.")
parser.add_argument(
    "--test_set",
    default='',
    type=str,
    help="Path to a testing set.")
parser.add_argument(
    "--model_path",
    default='',
    type=str,
    help="Path where to save the trained model.")
parser.add_argument(
    "--epochs",
    default=10,
    type=int,
    help="Number of epochs.")
parser.add_argument(
    "--embed_dim",
    default=8,
    type=int,
    help="Token embedding dimension.")
parser.add_argument(
    "--threads",
    default=1,
    type=int,
    help="Maximum number of threads to use.")
parser.add_argument(
    "--logdir",
    default='',
    type=str,
    help="Logdir.")
args = parser.parse_args()


SYMBOLS_WITH_ARITIES = {
    '+': 2,
    '-': 2
}

labels_train, inputs_train = load_data(args.train_set)
labels_valid, inputs_valid = load_data(args.valid_set)
modulo = max(i.item() for i in labels_train + labels_valid) + 1
numbers = set(''.join(inputs_train + inputs_valid)) - set(SYMBOLS_WITH_ARITIES)
consts_as_tensors = consts_to_tensors(numbers)
modules = instanciate_modules(SYMBOLS_WITH_ARITIES, len(numbers), modulo)
params_of_modules = parameters_of_modules(modules)
loss_1 = loss
optim_1 = torch.optim.SGD(params_of_modules, lr=0.001, momentum=0.7)
for e in range(args.epochs):
    train(inputs_train, labels_train, modules, loss_1, optim_1)
    acc = accuracy(inputs_valid, labels_valid, modules)
    print("Epoch: {}. Accuracy on validation: {}".format(e, acc))

