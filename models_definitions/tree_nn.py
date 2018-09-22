import torch
import sys
sys.path.append('.')


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
            torch.nn.Linear(dim_h, 2),
            # TODO maybe better 1 and sigmoid; but not with CrossEntropyLoss()
        )

    def forward(self, x): # TODO this repeats; make a super-class
        return self.model(x)


class TreeNN:
    def __init__(
        self,
        vocab,
        optimizer='SGD',
        learning_rate=0.001,
        momentum=0.8,
        loss=torch.nn.CrossEntropyLoss()
    ):
        # TODO add parameter for controlling layer size

        self.functs_with_arits = vocab.functs_with_arits
        self.vars_consts_as_torch_tensors = vocab.vars_consts_as_torch_tensors
        self.n_vars_consts = len(self.vars_consts_as_torch_tensors)

        # instantiate modules for all function symbols
        self.modules = {}
        for fun in self.functs_with_arits:
            self.modules[fun] = FunctionNetwork(fun, self.functs_with_arits[fun])
        self.modules['VARCONST'] = VarConstNetwork(dim_in=self.n_vars_consts)
        self.modules['EQUALITY'] = EqualityNetwork()
        self.loss = loss
        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=learning_rate,
                momentum=momentum)


    def parameters(self):
        parameters = []
        for m in self.modules:
            parameters.extend(self.modules[m].parameters())
        return parameters


    def tree(self, term):
        if len(term) > 1:
            x = torch.cat([self.tree(t) for t in term[1]], -1)
            return self.modules[term[0]](x)
        else:
            return self.modules['VARCONST'](
                self.vars_consts_as_torch_tensors[term[0]])


    def forward(self, term_L, term_R):
        return self.modules['EQUALITY'](torch.cat(
            [self.tree(term_L), self.tree(term_R)], -1))


    def train_one_example(self, example_with_label):
        label, term_L, term_R = example_with_label
        pred = self.forward(term_L, term_R)
        loss = self.loss(pred, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), pred.argmax().item()


    def train(self, train_set, epochs=10, verbose=True):
        for epoch in range(epochs):
            losses = []
            preds = []
            for example in train_set:
                loss, pred = self.train_one_example(example)
                losses.append(loss)
                preds.append(pred)
            N = len(train_set)
            labels = [e[0].item() for e in train_set]
            loss_avg = sum(losses) / N
            accuracy = sum(preds[i] == labels[i] for i in range(N)) / N
            if verbose:
                print("Epoch: {} Loss: {} Accuracy: {}.".format(
                    epoch, loss_avg, accuracy))


    def predict(self, examples):
        return [self.forward(e).argmax().item() for e in examples]



if __name__ == "__main__":
    import argparse
    import random
    from utils.dataset import DatasetTreeNN, VocabTreeNN


    random.seed(541)
    torch.random.manual_seed(541)


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
        "--functs_with_arits",
        type=str,
        help="Path to the file with function symbols and its arities.")
    parser.add_argument(
        "--vars_consts",
        type=str,
        help="Path to the file with constants and variables.")
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

    train_set = DatasetTreeNN(args.train_set)
    valid_set = DatasetTreeNN(args.valid_set)
    vocab = VocabTreeNN(args.functs_with_arits, args.vars_consts)
    net = TreeNN(vocab)

###### TEST ########################################################
    e = list(train_set)[0]
    print(e)
    print(net.forward(e[1], e[2]))
    print(net.loss(net.forward(e[2], e[1]), torch.tensor([1])))
    net.train_one_example(e)
    net.train(train_set)


