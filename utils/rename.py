#!/usr/bin/env python3
# usage: python3 utils/rename.py data/equiv.test --variables 'XYZUW'
import argparse, random
import itertools
import re

def variables(formula):
    '''
    Returns a set of all variables (X, Y, ...) appearing in a formula.
    '''
    return ''.join({i for i in formula if 'A' <= i <= 'Z'})

def rename(formula, available_variables):
    formula_curly_variables = re.sub(r'([A-Z])', r'{\1}', formula)
    original_variables = variables(formula)
    random_variables = random.sample(available_variables, len(original_variables))
    random_renaming = dict(zip(original_variables, random_variables))
    return formula_curly_variables.format(**random_renaming)

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str,
                    help="Path to a file with data to rename.")
parser.add_argument('--variables', type=str, default='XYZUW',
                    help="List of variables available for substituting.")
args = parser.parse_args()

with open(args.filename, 'r') as f:
    examples = f.read().splitlines()

for e in examples:
    print(rename(e, args.variables))
