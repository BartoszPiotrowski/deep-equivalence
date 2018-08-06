#!/usr/bin/env python3
# usage: python3 utils/augment.py data/equiv.train --variables 'XYZUW' --reverse
import argparse, random
import itertools
import re

def variables(formula):
    '''
    Returns a set of all variables (X, Y, ...) appearing in a formula.
    '''
    return ''.join({i for i in formula if 'A' <= i <= 'Z'})


def substitute(formula, substitution):
    '''
    formula is a string, substitution is a dictionary, like {'X': 'A', 'Y': 'B'}
    '''
    formula_curly_variables = re.sub(r'([A-Z])', r'{\1}', formula)
    return formula_curly_variables.format(**substitution)

def all_substitutions(original_formula, available_variables):
    '''
    Given a formula (string), returns a list of all possible injective
    substitutions of this formula with available variables.
    '''
    original_variables = variables(original_formula)
    all_substitutions = itertools.permutations(available_variables,
                                               len(original_variables))
    return [substitute(original_formula,
                       dict(zip(original_variables, substituted_variables))) \
                                for substituted_variables in all_substitutions]

def reverse(example):
    e = example.split(' ')
    return ' '.join([e[0], e[2], e[1]])

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str,
                    help="Path to a file with data to augment.")
parser.add_argument('--variables', type=str, default='XYZUW',
                    help="List of variables available for substituting.")
parser.add_argument('--reverse', type=bool, default=True,
                    help="Test set size.")
args = parser.parse_args()

with open(args.filename, 'r') as f:
    examples = f.read().splitlines()

augmented_examples = []
for e in examples:
    substitutions = all_substitutions(e, args.variables)
    for s in substitutions:
        augmented_examples.append(s)
        if args.reverse:
            augmented_examples.append(reverse(s))

random.shuffle(augmented_examples)

for e in augmented_examples:
    print(e)
